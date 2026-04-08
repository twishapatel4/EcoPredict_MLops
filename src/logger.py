import logging
import os
from datetime import datetime

# 1. Create a filename based on the current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# 2. Define the path where logs will be stored
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# 3. Create the directory if it doesn't exist
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# 4. Configure the logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Optional: To test if logger is working when running this file directly
if __name__ == "__main__":
    logging.info("Logging has started successfully.")