import logging
import os
from datetime import datetime

PROJECT_NAME = "Meli-Fraud-Detection"

# Create timestamp for log filename
LOG_FILE = f"{PROJECT_NAME}_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.log"

# Create logs directory if it doesn't exist
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Full path for the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging with detailed format
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d - %(name)s - %(levelname)s - %(message)s [%(pathname)s]",
    level=logging.INFO,
)
