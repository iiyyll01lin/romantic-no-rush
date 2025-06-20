import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    # Service Configuration
    PORT: int = 8753
    RELOAD: bool = True
    LOG_LEVEL: str = "INFO"
    
    # File Processing
    MAX_FILE_SIZE_BYTES: int = 100 * 1024 * 1024  # 100MB
    UPLOAD_DIR: str = "/app/uploads"
    RESULTS_DIR: str = "/app/results"
    TEMP_DIR: str = "/app/temp"
    
    # Document Processing
    SUPPORTED_FORMATS: list = ["pdf", "docx", "pptx", "xlsx", "txt", "html"]
    DEFAULT_LANGUAGE: str = "en"
    OCR_LANGUAGES: list = ["en", "zh", "ja", "ko", "fr", "de", "es"]
    
    # Magic-PDF Configuration
    MAGIC_PDF_MODEL_PATH: str = "/app/models"
    MAGIC_PDF_OUTPUT_DIR: str = "/app/results/magic_pdf"
    
    # Task Management
    MAX_CONCURRENT_TASKS: int = 5
    TASK_TIMEOUT_SECONDS: int = 600  # 10 minutes
    CLEANUP_INTERVAL_HOURS: int = 24
    RESULT_RETENTION_DAYS: int = 7
    
    # External Services
    DOC_INGESTER_URL: str = "http://doc-ingester:8752"
    YY_CHUNKER_URL: str = "http://yy-chunker:8754"
    
    # Database
    REDIS_URL: str = "redis://redis:6379"
    POSTGRES_URL: str = "postgresql://user:password@postgres:5432/deaap"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
