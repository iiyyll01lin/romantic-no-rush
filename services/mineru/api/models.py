import os
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    TXT = "txt"
    HTML = "html"


class OutputFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    TEXT = "text"


class ProcessingStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingRequest(BaseModel):
    filename: str = Field(..., description="Original filename")
    output_format: OutputFormat = Field(default=OutputFormat.MARKDOWN)
    extract_images: bool = Field(default=True)
    extract_tables: bool = Field(default=True)
    ocr_enabled: bool = Field(default=True)
    language: str = Field(default="en")
    quality: str = Field(default="high", description="Processing quality: low, medium, high")
    
    class Config:
        use_enum_values = True


class ProcessingResponse(BaseModel):
    task_id: str
    status: ProcessingStatus
    message: str
    estimated_completion: Optional[str] = None
    
    class Config:
        use_enum_values = True


class ExtractionResult(BaseModel):
    task_id: str
    status: ProcessingStatus
    filename: str
    output_format: OutputFormat
    markdown_content: Optional[str] = None
    extracted_images: Optional[list] = None
    extracted_tables: Optional[list] = None
    metadata: Optional[dict] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    created_at: str
    completed_at: Optional[str] = None
    
    class Config:
        use_enum_values = True


class TaskStatus(BaseModel):
    task_id: str
    status: ProcessingStatus
    progress: int = Field(ge=0, le=100)
    current_step: str
    total_steps: int
    estimated_completion: Optional[str] = None
    error_message: Optional[str] = None
    
    class Config:
        use_enum_values = True
