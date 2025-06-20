#!/usr/bin/env python3
"""
DEAAP LLM Runtime Service
Serves LLM models with LORA adaptor support for agent deployment
"""

import asyncio
import json
import logging
import os
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, AsyncIterator
from dataclasses import dataclass
from enum import Enum

import aioredis
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean, Integer, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
    BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
)
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model, TaskType
import accelerate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://deaap:deaap@postgres:5432/deaap")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/models")
ADAPTOR_CACHE_DIR = os.getenv("ADAPTOR_CACHE_DIR", "/app/adaptors")
BASE_MODELS = os.getenv("BASE_MODELS", "llama-2-7b-chat-hf,codellama-7b-instruct-hf").split(",")
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "4096"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
INFERENCE_ENGINE = os.getenv("INFERENCE_ENGINE", "transformers")  # transformers, vllm
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_IN_8BIT = os.getenv("LOAD_IN_8BIT", "true").lower() == "true"

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ModelStatus(Enum):
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UNLOADED = "unloaded"

class InferenceRequest(Base):
    __tablename__ = "inference_requests"
    
    id = Column(String, primary_key=True)
    model_name = Column(String, nullable=False)
    adaptor_name = Column(String)
    prompt = Column(Text, nullable=False)
    parameters = Column(JSON)
    response = Column(Text)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    processing_time = Column(Float)
    tokens_generated = Column(Integer)

class LoadedModel(Base):
    __tablename__ = "loaded_models"
    
    id = Column(String, primary_key=True)
    model_name = Column(String, nullable=False)
    adaptor_name = Column(String)
    status = Column(String, default=ModelStatus.LOADING.value)
    loaded_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, default=datetime.utcnow)
    memory_usage = Column(Float)
    parameters = Column(JSON)

# Create tables
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI(title="DEAAP LLM Runtime Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model_manager = None
redis_client = None

# Pydantic models
class InferenceParameters(BaseModel):
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=100)
    do_sample: bool = Field(default=True)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    num_beams: int = Field(default=1, ge=1, le=8)
    early_stopping: bool = Field(default=True)
    pad_token_id: Optional[int] = Field(default=None)
    eos_token_id: Optional[int] = Field(default=None)

class InferenceRequestModel(BaseModel):
    prompt: str = Field(..., description="Input prompt for the model")
    model: str = Field(..., description="Model name to use")
    adaptor: Optional[str] = Field(default=None, description="LORA adaptor to apply")
    parameters: InferenceParameters = Field(default_factory=InferenceParameters)
    stream: bool = Field(default=False, description="Enable streaming response")
    system_prompt: Optional[str] = Field(default=None, description="System prompt for chat models")
    chat_template: bool = Field(default=True, description="Use chat template formatting")

class InferenceResponse(BaseModel):
    request_id: str
    model_used: str
    adaptor_used: Optional[str]
    response: str
    processing_time: float
    tokens_generated: int
    finish_reason: str
    metadata: Optional[Dict[str, Any]] = None

class ModelLoadRequest(BaseModel):
    model_name: str = Field(..., description="Model name to load")
    adaptor_name: Optional[str] = Field(default=None, description="LORA adaptor to load with model")
    load_in_8bit: bool = Field(default=LOAD_IN_8BIT, description="Load model in 8-bit precision")
    trust_remote_code: bool = Field(default=False, description="Trust remote code for model loading")

class ModelInfo(BaseModel):
    model_name: str
    adaptor_name: Optional[str]
    status: str
    loaded_at: datetime
    last_used: datetime
    memory_usage: float
    parameters: Dict[str, Any]

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Custom stopping criteria
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# Model management
class LLMModelManager:
    def __init__(self):
        self.models = {}  # {model_key: {"model": model, "tokenizer": tokenizer, "config": config}}
        self.adaptors = {}  # {adaptor_name: adaptor_path}
        
    def _get_model_key(self, model_name: str, adaptor_name: str = None) -> str:
        """Generate unique key for model + adaptor combination"""
        return f"{model_name}:{adaptor_name or 'base'}"
    
    async def load_model(self, model_name: str, adaptor_name: str = None, 
                        load_in_8bit: bool = LOAD_IN_8BIT, trust_remote_code: bool = False) -> Dict[str, Any]:
        """Load a model with optional LORA adaptor"""
        model_key = self._get_model_key(model_name, adaptor_name)
        
        if model_key in self.models:
            logger.info(f"Model {model_key} already loaded")
            return self.models[model_key]
        
        logger.info(f"Loading model: {model_name}" + (f" with adaptor: {adaptor_name}" if adaptor_name else ""))
        
        try:
            # Configure quantization
            quantization_config = None
            if load_in_8bit and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    llm_int8_enable_fp32_cpu_offload=True
                )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=MODEL_CACHE_DIR,
                trust_remote_code=trust_remote_code,
                padding_side="left"
            )
            
            # Add pad token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=MODEL_CACHE_DIR,
                quantization_config=quantization_config,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            # Apply LORA adaptor if specified
            if adaptor_name:
                adaptor_path = os.path.join(ADAPTOR_CACHE_DIR, adaptor_name)
                if os.path.exists(adaptor_path):
                    logger.info(f"Loading LORA adaptor from: {adaptor_path}")
                    model = PeftModel.from_pretrained(model, adaptor_path)
                else:
                    logger.warning(f"Adaptor not found: {adaptor_path}")
                    adaptor_name = None
            
            # Prepare model for inference
            model.eval()
            if hasattr(model, 'generation_config'):
                model.generation_config.pad_token_id = tokenizer.pad_token_id
            
            # Calculate memory usage
            memory_usage = 0.0
            if torch.cuda.is_available():
                memory_usage = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            model_info = {
                "model": model,
                "tokenizer": tokenizer,
                "config": {
                    "model_name": model_name,
                    "adaptor_name": adaptor_name,
                    "load_in_8bit": load_in_8bit,
                    "memory_usage": memory_usage,
                    "vocab_size": tokenizer.vocab_size,
                    "max_position_embeddings": getattr(model.config, 'max_position_embeddings', MAX_SEQUENCE_LENGTH)
                }
            }
            
            self.models[model_key] = model_info
            logger.info(f"Successfully loaded model: {model_key} ({memory_usage:.2f} GB)")
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    
    def get_model(self, model_name: str, adaptor_name: str = None) -> Optional[Dict[str, Any]]:
        """Get loaded model info"""
        model_key = self._get_model_key(model_name, adaptor_name)
        return self.models.get(model_key)
    
    def unload_model(self, model_name: str, adaptor_name: str = None):
        """Unload a model to free memory"""
        model_key = self._get_model_key(model_name, adaptor_name)
        if model_key in self.models:
            del self.models[model_key]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded model: {model_key}")
            return True
        return False
    
    def list_loaded_models(self) -> List[str]:
        """List all loaded models"""
        return list(self.models.keys())
    
    async def generate_text(self, prompt: str, model_name: str, adaptor_name: str = None,
                          parameters: InferenceParameters = InferenceParameters(),
                          system_prompt: str = None, chat_template: bool = True) -> Dict[str, Any]:
        """Generate text using a loaded model"""
        model_info = self.get_model(model_name, adaptor_name)
        if not model_info:
            model_info = await self.load_model(model_name, adaptor_name)
        
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        
        start_time = time.time()
        
        try:
            # Format prompt with chat template if requested
            formatted_prompt = prompt
            if chat_template and system_prompt and hasattr(tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                try:
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except:
                    # Fallback to simple format
                    formatted_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            elif system_prompt:
                formatted_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            
            # Tokenize input
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH - parameters.max_new_tokens
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Prepare generation config
            generation_config = GenerationConfig(
                max_new_tokens=parameters.max_new_tokens,
                temperature=parameters.temperature,
                top_p=parameters.top_p,
                top_k=parameters.top_k,
                do_sample=parameters.do_sample,
                repetition_penalty=parameters.repetition_penalty,
                num_beams=parameters.num_beams,
                early_stopping=parameters.early_stopping,
                pad_token_id=parameters.pad_token_id or tokenizer.pad_token_id,
                eos_token_id=parameters.eos_token_id or tokenizer.eos_token_id,
                use_cache=True
            )
            
            # Set up stopping criteria
            stop_tokens = [tokenizer.eos_token_id]
            if tokenizer.pad_token_id != tokenizer.eos_token_id:
                stop_tokens.append(tokenizer.pad_token_id)
            
            stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_tokens)])
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    stopping_criteria=stopping_criteria,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Decode response
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs.sequences[0][input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            processing_time = time.time() - start_time
            
            # Determine finish reason
            finish_reason = "stop"
            if len(generated_tokens) >= parameters.max_new_tokens:
                finish_reason = "length"
            
            return {
                "response": response.strip(),
                "processing_time": processing_time,
                "tokens_generated": len(generated_tokens),
                "finish_reason": finish_reason,
                "model_used": model_name,
                "adaptor_used": adaptor_name
            }
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Text generation failed: {e}")

# Global model manager
model_manager = LLMModelManager()

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    global redis_client
    logger.info("Starting DEAAP LLM Runtime Service...")
    
    # Initialize Redis
    redis_client = aioredis.from_url(REDIS_URL)
    
    # Load default model if specified
    if BASE_MODELS and BASE_MODELS[0]:
        try:
            await model_manager.load_model(BASE_MODELS[0])
        except Exception as e:
            logger.warning(f"Failed to load default model: {e}")
    
    logger.info("LLM Runtime Service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    global redis_client
    if redis_client:
        await redis_client.close()
    logger.info("LLM Runtime Service shut down")

# API endpoints
@app.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequestModel, db: Session = Depends(get_db)):
    """Generate text using specified model and adaptor"""
    request_id = f"gen-{int(time.time())}-{hashlib.md5(request.prompt.encode()).hexdigest()[:8]}"
    
    # Store request
    inference_request = InferenceRequest(
        id=request_id,
        model_name=request.model,
        adaptor_name=request.adaptor,
        prompt=request.prompt,
        parameters=request.parameters.dict(),
        status="processing"
    )
    
    db.add(inference_request)
    db.commit()
    
    try:
        # Generate text
        result = await model_manager.generate_text(
            prompt=request.prompt,
            model_name=request.model,
            adaptor_name=request.adaptor,
            parameters=request.parameters,
            system_prompt=request.system_prompt,
            chat_template=request.chat_template
        )
        
        # Update request record
        inference_request.response = result["response"]
        inference_request.status = "completed"
        inference_request.completed_at = datetime.utcnow()
        inference_request.processing_time = result["processing_time"]
        inference_request.tokens_generated = result["tokens_generated"]
        db.commit()
        
        return InferenceResponse(
            request_id=request_id,
            model_used=result["model_used"],
            adaptor_used=result["adaptor_used"],
            response=result["response"],
            processing_time=result["processing_time"],
            tokens_generated=result["tokens_generated"],
            finish_reason=result["finish_reason"]
        )
        
    except Exception as e:
        inference_request.status = "error"
        inference_request.completed_at = datetime.utcnow()
        db.commit()
        raise e

@app.post("/generate/stream")
async def stream_generate_text(request: InferenceRequestModel):
    """Generate text with streaming response"""
    # Note: Streaming implementation would require more complex generator setup
    # For now, return as non-streaming
    result = await generate_text(request)
    
    async def generate():
        yield f"data: {json.dumps(result.dict())}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/models/load")
async def load_model(request: ModelLoadRequest, db: Session = Depends(get_db)):
    """Load a model with optional LORA adaptor"""
    try:
        model_info = await model_manager.load_model(
            model_name=request.model_name,
            adaptor_name=request.adaptor_name,
            load_in_8bit=request.load_in_8bit,
            trust_remote_code=request.trust_remote_code
        )
        
        # Store in database
        model_key = model_manager._get_model_key(request.model_name, request.adaptor_name)
        loaded_model = LoadedModel(
            id=model_key,
            model_name=request.model_name,
            adaptor_name=request.adaptor_name,
            status=ModelStatus.READY.value,
            memory_usage=model_info["config"]["memory_usage"],
            parameters=model_info["config"]
        )
        
        db.merge(loaded_model)
        db.commit()
        
        return {
            "message": f"Model loaded successfully",
            "model_name": request.model_name,
            "adaptor_name": request.adaptor_name,
            "memory_usage": model_info["config"]["memory_usage"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

@app.delete("/models/{model_name}")
async def unload_model(model_name: str, adaptor_name: Optional[str] = None, db: Session = Depends(get_db)):
    """Unload a model to free memory"""
    success = model_manager.unload_model(model_name, adaptor_name)
    
    if success:
        # Update database
        model_key = model_manager._get_model_key(model_name, adaptor_name)
        loaded_model = db.query(LoadedModel).filter(LoadedModel.id == model_key).first()
        if loaded_model:
            loaded_model.status = ModelStatus.UNLOADED.value
            db.commit()
        
        return {"message": f"Model {model_key} unloaded successfully"}
    else:
        raise HTTPException(status_code=404, detail="Model not found")

@app.get("/models", response_model=List[ModelInfo])
async def list_models(db: Session = Depends(get_db)):
    """List all loaded models"""
    loaded_models = db.query(LoadedModel).filter(LoadedModel.status == ModelStatus.READY.value).all()
    
    return [
        ModelInfo(
            model_name=model.model_name,
            adaptor_name=model.adaptor_name,
            status=model.status,
            loaded_at=model.loaded_at,
            last_used=model.last_used,
            memory_usage=model.memory_usage,
            parameters=model.parameters or {}
        )
        for model in loaded_models
    ]

@app.get("/models/available")
async def list_available_models():
    """List available base models and adaptors"""
    # Scan for available models and adaptors
    available_models = BASE_MODELS.copy()
    
    adaptors = []
    if os.path.exists(ADAPTOR_CACHE_DIR):
        adaptors = [d for d in os.listdir(ADAPTOR_CACHE_DIR) 
                   if os.path.isdir(os.path.join(ADAPTOR_CACHE_DIR, d))]
    
    return {
        "base_models": available_models,
        "available_adaptors": adaptors,
        "loaded_models": model_manager.list_loaded_models(),
        "inference_engine": INFERENCE_ENGINE,
        "device": DEVICE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_available": True,
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "memory_allocated": torch.cuda.memory_allocated() / 1024**3,
            "memory_cached": torch.cuda.memory_reserved() / 1024**3
        }
    
    return {
        "status": "healthy",
        "service": "llm-runtime",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "loaded_models": len(model_manager.models),
        "inference_engine": INFERENCE_ENGINE,
        "device": DEVICE,
        "gpu_info": gpu_info
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8772)
