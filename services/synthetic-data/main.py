#!/usr/bin/env python3
"""
DEAAP Synthetic Data Service
Generates synthetic training datasets from document chunks for LLM fine-tuning
"""

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import aioredis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import openai
from transformers import pipeline, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MAX_SYNTHETIC_PAIRS = int(os.getenv("MAX_SYNTHETIC_PAIRS", "1000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
QUALITY_THRESHOLD = float(os.getenv("QUALITY_THRESHOLD", "0.7"))

# Set OpenAI API key if available
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

app = FastAPI(
    title="DEAAP Synthetic Data Service",
    description="Generate synthetic training datasets from document chunks",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChunkInput(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    chunk_id: str = ""

class SyntheticDataRequest(BaseModel):
    chunks: List[ChunkInput]
    dataset_type: str = Field(default="instruction", description="Type of synthetic data: instruction, qa, completion")
    pairs_per_chunk: int = Field(default=3, ge=1, le=10)
    quality_filter: bool = Field(default=True)
    output_format: str = Field(default="jsonl", description="Output format: jsonl, csv, json")
    use_llm: bool = Field(default=True, description="Use LLM for generation or rule-based")

class SyntheticPair(BaseModel):
    instruction: str
    input: str = ""
    output: str
    source_chunk: str
    quality_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SyntheticDataResponse(BaseModel):
    task_id: str
    status: str
    total_pairs: int
    high_quality_pairs: int
    dataset_url: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TaskStatus(BaseModel):
    task_id: str
    status: str  # processing, completed, failed
    progress: float
    total_chunks: int
    processed_chunks: int
    generated_pairs: int
    high_quality_pairs: int
    error_message: str = ""
    started_at: datetime
    completed_at: Optional[datetime] = None

# Global task storage
active_tasks: Dict[str, TaskStatus] = {}

# Redis client
redis_client = None

async def get_redis():
    global redis_client
    if redis_client is None:
        redis_client = await aioredis.from_url(REDIS_URL)
    return redis_client

class SyntheticDataGenerator:
    """
    Core synthetic data generation logic
    """
    
    def __init__(self):
        self.templates = {
            "instruction": [
                "Explain the concept mentioned in the following text: {text}",
                "What are the key points discussed in this passage: {text}",
                "Summarize the main ideas from: {text}",
                "Based on this text, what would be the most important takeaway: {text}",
                "How would you explain this to a non-expert: {text}"
            ],
            "qa": [
                "Question: What is the main topic of this text?\nContext: {text}",
                "Question: What are the key findings mentioned?\nContext: {text}",
                "Question: How does this relate to the broader subject?\nContext: {text}",
                "Question: What evidence supports the claims made?\nContext: {text}",
                "Question: What are the practical implications?\nContext: {text}"
            ],
            "completion": [
                "Complete this explanation: {text_partial}",
                "Continue this analysis: {text_partial}",
                "Finish this description: {text_partial}",
                "Extend this discussion: {text_partial}"
            ]
        }
        
        # Initialize tokenizer for quality scoring
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
            self.tokenizer = None
    
    async def generate_synthetic_pairs(
        self, 
        chunks: List[ChunkInput], 
        dataset_type: str,
        pairs_per_chunk: int,
        use_llm: bool = True
    ) -> List[SyntheticPair]:
        """
        Generate synthetic training pairs from chunks
        """
        all_pairs = []
        
        for chunk in chunks:
            try:
                if use_llm and OPENAI_API_KEY:
                    pairs = await self._generate_with_llm(chunk, dataset_type, pairs_per_chunk)
                else:
                    pairs = await self._generate_rule_based(chunk, dataset_type, pairs_per_chunk)
                
                all_pairs.extend(pairs)
                
            except Exception as e:
                logger.error(f"Error generating pairs for chunk {chunk.chunk_id}: {e}")
                continue
        
        return all_pairs
    
    async def _generate_with_llm(
        self, 
        chunk: ChunkInput, 
        dataset_type: str, 
        pairs_per_chunk: int
    ) -> List[SyntheticPair]:
        """
        Generate synthetic pairs using LLM
        """
        pairs = []
        
        try:
            # Create system prompt based on dataset type
            if dataset_type == "instruction":
                system_prompt = """Generate diverse instruction-following examples based on the provided text. 
                Create instructions that would help someone learn about the content.
                Return JSON format with 'instruction', 'input', and 'output' fields."""
            elif dataset_type == "qa":
                system_prompt = """Generate question-answer pairs based on the provided text.
                Create questions that test understanding of the content.
                Return JSON format with 'question', 'context', and 'answer' fields."""
            else:  # completion
                system_prompt = """Generate text completion examples based on the provided text.
                Create partial text that can be completed with the remaining content.
                Return JSON format with 'prompt' and 'completion' fields."""
            
            # Call OpenAI API
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Text: {chunk.text[:2000]}"}  # Limit text length
                ],
                max_tokens=1000,
                temperature=0.8
            )
            
            # Parse response and create pairs
            content = response.choices[0].message.content
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data[:pairs_per_chunk]:
                        pair = self._create_pair_from_llm_output(item, chunk, dataset_type)
                        if pair:
                            pairs.append(pair)
                else:
                    pair = self._create_pair_from_llm_output(data, chunk, dataset_type)
                    if pair:
                        pairs.append(pair)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse LLM response as JSON for chunk {chunk.chunk_id}")
                
        except Exception as e:
            logger.error(f"LLM generation failed for chunk {chunk.chunk_id}: {e}")
            # Fallback to rule-based generation
            pairs = await self._generate_rule_based(chunk, dataset_type, pairs_per_chunk)
        
        return pairs
    
    def _create_pair_from_llm_output(
        self, 
        data: Dict[str, Any], 
        chunk: ChunkInput, 
        dataset_type: str
    ) -> Optional[SyntheticPair]:
        """
        Create SyntheticPair from LLM output
        """
        try:
            if dataset_type == "instruction":
                instruction = data.get("instruction", "")
                input_text = data.get("input", "")
                output = data.get("output", "")
            elif dataset_type == "qa":
                instruction = data.get("question", "")
                input_text = data.get("context", "")
                output = data.get("answer", "")
            else:  # completion
                instruction = "Complete the following text:"
                input_text = data.get("prompt", "")
                output = data.get("completion", "")
            
            if instruction and output:
                quality_score = self._calculate_quality_score(instruction, input_text, output)
                
                return SyntheticPair(
                    instruction=instruction,
                    input=input_text,
                    output=output,
                    source_chunk=chunk.chunk_id,
                    quality_score=quality_score,
                    metadata={
                        "generation_method": "llm",
                        "source_metadata": chunk.metadata,
                        "dataset_type": dataset_type
                    }
                )
        except Exception as e:
            logger.error(f"Error creating pair from LLM output: {e}")
        
        return None
    
    async def _generate_rule_based(
        self, 
        chunk: ChunkInput, 
        dataset_type: str, 
        pairs_per_chunk: int
    ) -> List[SyntheticPair]:
        """
        Generate synthetic pairs using rule-based templates
        """
        pairs = []
        templates = self.templates.get(dataset_type, self.templates["instruction"])
        
        # Select random templates
        selected_templates = random.sample(templates, min(pairs_per_chunk, len(templates)))
        
        for template in selected_templates:
            try:
                if dataset_type == "completion":
                    # For completion, use part of the text as input
                    text_words = chunk.text.split()
                    if len(text_words) > 20:
                        split_point = len(text_words) // 2
                        input_text = " ".join(text_words[:split_point])
                        output = " ".join(text_words[split_point:])
                        instruction = template.format(text_partial=input_text)
                    else:
                        continue
                else:
                    # For instruction and QA, use full text as context
                    if "{text}" in template:
                        instruction = template.format(text=chunk.text[:500])  # Limit length
                        input_text = ""
                        output = self._generate_response_for_template(template, chunk.text)
                    else:
                        continue
                
                quality_score = self._calculate_quality_score(instruction, input_text, output)
                
                pair = SyntheticPair(
                    instruction=instruction,
                    input=input_text,
                    output=output,
                    source_chunk=chunk.chunk_id,
                    quality_score=quality_score,
                    metadata={
                        "generation_method": "rule_based",
                        "template": template,
                        "source_metadata": chunk.metadata,
                        "dataset_type": dataset_type
                    }
                )
                pairs.append(pair)
                
            except Exception as e:
                logger.error(f"Error in rule-based generation: {e}")
                continue
        
        return pairs
    
    def _generate_response_for_template(self, template: str, text: str) -> str:
        """
        Generate appropriate response based on template type
        """
        if "explain" in template.lower():
            return f"The text discusses {text[:200]}... This relates to key concepts and provides insights into the subject matter."
        elif "key points" in template.lower():
            sentences = text.split('.')[:3]
            return f"The main points are: {'. '.join(sentences)}."
        elif "summarize" in template.lower():
            return f"Summary: {text[:300]}..."
        elif "takeaway" in template.lower():
            return f"The most important takeaway is that {text.split('.')[0]}."
        elif "question:" in template.lower():
            return f"Based on the context provided, {text.split('.')[0]}."
        else:
            return f"This text covers: {text[:200]}..."
    
    def _calculate_quality_score(self, instruction: str, input_text: str, output: str) -> float:
        """
        Calculate quality score for synthetic pair
        """
        score = 0.5  # Base score
        
        # Check length appropriateness
        if 10 <= len(instruction.split()) <= 50:
            score += 0.1
        if 20 <= len(output.split()) <= 200:
            score += 0.1
        
        # Check for repetition
        instruction_words = set(instruction.lower().split())
        output_words = set(output.lower().split())
        overlap = len(instruction_words.intersection(output_words))
        if overlap < len(instruction_words) * 0.5:  # Less than 50% word overlap
            score += 0.1
        
        # Check for meaningfulness (simple heuristics)
        if any(word in output.lower() for word in ["the", "and", "or", "but", "however", "therefore"]):
            score += 0.1
        
        # Penalize generic responses
        generic_phrases = ["this text", "the passage", "as mentioned", "according to"]
        if not any(phrase in output.lower() for phrase in generic_phrases):
            score += 0.1
        
        return min(score, 1.0)
    
    def filter_by_quality(self, pairs: List[SyntheticPair], threshold: float) -> List[SyntheticPair]:
        """
        Filter pairs by quality threshold
        """
        return [pair for pair in pairs if pair.quality_score >= threshold]

# Initialize generator
generator = SyntheticDataGenerator()

@app.post("/generate", response_model=SyntheticDataResponse)
async def generate_synthetic_data(
    request: SyntheticDataRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate synthetic training data from document chunks
    """
    task_id = f"synthetic_{int(time.time())}_{random.randint(1000, 9999)}"
    
    # Initialize task status
    task_status = TaskStatus(
        task_id=task_id,
        status="processing",
        progress=0.0,
        total_chunks=len(request.chunks),
        processed_chunks=0,
        generated_pairs=0,
        high_quality_pairs=0,
        started_at=datetime.utcnow()
    )
    active_tasks[task_id] = task_status
    
    # Start background processing
    background_tasks.add_task(
        process_synthetic_generation,
        task_id,
        request
    )
    
    return SyntheticDataResponse(
        task_id=task_id,
        status="processing",
        total_pairs=0,
        high_quality_pairs=0,
        metadata={
            "total_chunks": len(request.chunks),
            "dataset_type": request.dataset_type,
            "pairs_per_chunk": request.pairs_per_chunk
        }
    )

async def process_synthetic_generation(task_id: str, request: SyntheticDataRequest):
    """
    Background task for synthetic data generation
    """
    try:
        task = active_tasks[task_id]
        
        # Generate synthetic pairs
        all_pairs = await generator.generate_synthetic_pairs(
            request.chunks,
            request.dataset_type,
            request.pairs_per_chunk,
            request.use_llm
        )
        
        task.generated_pairs = len(all_pairs)
        task.progress = 0.7
        
        # Filter by quality if requested
        if request.quality_filter:
            high_quality_pairs = generator.filter_by_quality(all_pairs, QUALITY_THRESHOLD)
        else:
            high_quality_pairs = all_pairs
        
        task.high_quality_pairs = len(high_quality_pairs)
        task.progress = 0.9
        
        # Save dataset
        dataset_path = f"/app/data/synthetic/{task_id}.{request.output_format}"
        await save_synthetic_dataset(high_quality_pairs, dataset_path, request.output_format)
        
        # Update task status
        task.status = "completed"
        task.progress = 1.0
        task.completed_at = datetime.utcnow()
        
        # Store in Redis for retrieval
        redis = await get_redis()
        await redis.setex(
            f"synthetic_task:{task_id}",
            3600,  # 1 hour expiry
            json.dumps({
                "dataset_path": dataset_path,
                "total_pairs": len(all_pairs),
                "high_quality_pairs": len(high_quality_pairs),
                "metadata": {
                    "dataset_type": request.dataset_type,
                    "generation_method": "llm" if request.use_llm else "rule_based",
                    "quality_threshold": QUALITY_THRESHOLD if request.quality_filter else None
                }
            })
        )
        
        logger.info(f"Synthetic data generation completed for task {task_id}: {len(high_quality_pairs)} high-quality pairs")
        
    except Exception as e:
        logger.error(f"Synthetic data generation failed for task {task_id}: {e}")
        task = active_tasks.get(task_id)
        if task:
            task.status = "failed"
            task.error_message = str(e)

async def save_synthetic_dataset(
    pairs: List[SyntheticPair], 
    output_path: str, 
    format_type: str
):
    """
    Save synthetic dataset to file
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format_type == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(pair.dict()) + '\n')
    elif format_type == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([pair.dict() for pair in pairs], f, indent=2, ensure_ascii=False)
    elif format_type == "csv":
        import csv
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if pairs:
                writer = csv.DictWriter(f, fieldnames=pairs[0].dict().keys())
                writer.writeheader()
                for pair in pairs:
                    writer.writerow(pair.dict())

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """
    Get status of synthetic data generation task
    """
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return active_tasks[task_id]

@app.get("/download/{task_id}")
async def download_dataset(task_id: str):
    """
    Download generated synthetic dataset
    """
    redis = await get_redis()
    task_data = await redis.get(f"synthetic_task:{task_id}")
    
    if not task_data:
        raise HTTPException(status_code=404, detail="Dataset not found or expired")
    
    data = json.loads(task_data)
    dataset_path = data["dataset_path"]
    
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    # Return file for download
    from fastapi.responses import FileResponse
    return FileResponse(
        dataset_path,
        filename=f"synthetic_dataset_{task_id}.{dataset_path.split('.')[-1]}",
        media_type='application/octet-stream'
    )

@app.get("/templates")
async def get_templates():
    """
    Get available templates for different dataset types
    """
    return {
        "templates": generator.templates,
        "supported_types": list(generator.templates.keys()),
        "description": {
            "instruction": "Generate instruction-following training pairs",
            "qa": "Generate question-answer pairs for comprehension",
            "completion": "Generate text completion examples"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "synthetic-data",
        "timestamp": datetime.utcnow().isoformat(),
        "active_tasks": len(active_tasks),
        "llm_available": bool(OPENAI_API_KEY),
        "features": {
            "rule_based_generation": True,
            "llm_generation": bool(OPENAI_API_KEY),
            "quality_filtering": True,
            "multiple_formats": True
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8756,
        reload=True
    )
