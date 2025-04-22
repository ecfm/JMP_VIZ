import os
import logging
import datetime
from logging.handlers import RotatingFileHandler
import json

# Ensure the logs directory exists
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Get today's date for log filenames
today = datetime.datetime.now().strftime('%Y-%m-%d')

# Set up formatters
standard_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Custom JSON formatter for user logs
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        
        # Add additional fields if they exist
        if hasattr(record, 'user'):
            log_record['user'] = record.user
        if hasattr(record, 'path'):
            log_record['path'] = record.path
        if hasattr(record, 'action'):
            log_record['action'] = record.action
        if hasattr(record, 'context'):
            log_record['context'] = record.context
            
        return json.dumps(log_record, ensure_ascii=False)

# Configure error logger
error_logger = logging.getLogger('error_logger')
error_logger.setLevel(logging.ERROR)
error_logger.propagate = False  # Prevent propagation to avoid duplicates
error_file_handler = RotatingFileHandler(
    os.path.join(logs_dir, f'error_{today}.log'),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
error_file_handler.setLevel(logging.ERROR)  # Set handler level explicitly
error_file_handler.setFormatter(standard_formatter)
error_logger.addHandler(error_file_handler)

# Configure application logger
app_logger = logging.getLogger('app_logger')
app_logger.setLevel(logging.INFO)
app_logger.propagate = False  # Prevent propagation to avoid duplicates
app_file_handler = RotatingFileHandler(
    os.path.join(logs_dir, f'app_{today}.log'),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
app_file_handler.setLevel(logging.INFO)  # Set handler level explicitly
app_file_handler.setFormatter(standard_formatter)
app_logger.addHandler(app_file_handler)

# Configure user action logger
user_logger = logging.getLogger('user_logger')
user_logger.setLevel(logging.INFO)
user_logger.propagate = False  # Prevent propagation to avoid duplicates
user_file_handler = RotatingFileHandler(
    os.path.join(logs_dir, f'user_{today}.log'),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
user_file_handler.setLevel(logging.INFO)  # Set handler level explicitly
user_file_handler.setFormatter(JsonFormatter())
user_logger.addHandler(user_file_handler)

# Configure root logger to capture uncaught exceptions
root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR)  # Only log errors at the root level
root_error_handler = RotatingFileHandler(  # Create a separate handler for root logger
    os.path.join(logs_dir, f'error_{today}.log'),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
root_error_handler.setLevel(logging.ERROR)  # Set handler level explicitly
root_error_handler.setFormatter(standard_formatter)
root_logger.addHandler(root_error_handler)  # Use the new handler for root

def log_error(message, exc_info=None):
    """Log an error message"""
    error_logger.error(message, exc_info=exc_info)

def log_app_event(message, level=logging.INFO):
    """Log an application event"""
    app_logger.log(level, message)

def log_user_action(user, path, action, context):
    """Log a user action in JSON format"""
    extra = {
        'user': user or 'Unknown',
        'path': path or 'Unknown',
        'action': action,
        'context': json.dumps(context, ensure_ascii=False, default=str)
    }
    user_logger.info('User action', extra=extra) 