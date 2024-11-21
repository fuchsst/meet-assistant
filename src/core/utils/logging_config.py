"""Logging configuration for Meeting Assistant."""
import logging
import logging.config
import json
from datetime import datetime
from pathlib import Path
from pythonjsonlogger import jsonlogger

from config.config import LOGS_DIR

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""
    
    def add_fields(self, log_record, record, message_dict):
        """Add custom fields to log record."""
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        
        # Add timestamp in ISO format
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add log level name
        log_record['level'] = record.levelname
        
        # Add module and function names
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        
        # Add process and thread IDs
        log_record['process_id'] = record.process
        log_record['thread_id'] = record.thread
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_record['correlation_id'] = record.correlation_id

def setup_logging():
    """Configure logging for the application."""
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Log file paths
    app_log = LOGS_DIR / "app.log"
    error_log = LOGS_DIR / "error.log"
    audit_log = LOGS_DIR / "audit.log"
    
    # Logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": CustomJsonFormatter,
                "format": "%(timestamp)s %(level)s %(name)s %(message)s"
            },
            "console": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "console",
                "level": "INFO"
            },
            "app_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(app_log),
                "formatter": "json",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "level": "DEBUG"
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(error_log),
                "formatter": "json",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "level": "ERROR"
            },
            "audit_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(audit_log),
                "formatter": "json",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "level": "INFO"
            }
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "app_file", "error_file"],
                "level": "DEBUG",
                "propagate": True
            },
            "audit": {  # Audit logger
                "handlers": ["audit_file"],
                "level": "INFO",
                "propagate": False
            }
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Create audit logger
    audit_logger = logging.getLogger("audit")
    
    return audit_logger

def log_audit(event_type: str, details: dict, user: str = "system"):
    """Log audit events."""
    audit_logger = logging.getLogger("audit")
    
    audit_event = {
        "event_type": event_type,
        "user": user,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details
    }
    
    audit_logger.info("Audit event", extra={"audit_event": audit_event})

# Example audit events
AUDIT_EVENTS = {
    "MEETING_CREATED": "Meeting created",
    "RECORDING_STARTED": "Recording started",
    "RECORDING_STOPPED": "Recording stopped",
    "ANALYSIS_GENERATED": "Analysis generated",
    "FILE_EXPORTED": "File exported",
    "BACKUP_CREATED": "Backup created",
    "MEETING_DELETED": "Meeting deleted",
    "CONFIG_CHANGED": "Configuration changed",
    "USER_ACTION": "User action performed",
    "SYSTEM_ERROR": "System error occurred"
}
