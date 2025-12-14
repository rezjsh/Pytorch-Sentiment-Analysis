import logging
import os
import sys

from core.singleton import SingletonMeta

class Logger(metaclass=SingletonMeta):
    """
    A singleton class for managing application-wide logging.
    It configures and provides a pre-configured logger instance using named handlers.
    """
    def __init__(self, logger_name="sentiment_analysis", log_dir="logs", log_file_name="running_logs.log"):
        """
        Initializes the Logger.

        Args:
            logger_name (str): The name of the logger to be used (e.g., 'sentiment_analysis').
            log_dir (str): The directory where log files will be stored.
            log_file_name (str): The name of the log file.
        """
        self._logger = logging.getLogger(logger_name) # Get the logger by name immediately
        
        #Prevent re-initialization if the logger already has handlers (means it's configured)
        if not self._logger.handlers:
            self._logger_name = logger_name
            self._log_dir = log_dir
            self._log_file_name = log_file_name
            self._logger.setLevel(logging.INFO) # Set default level here
            self._setup_logging()
            
    def _setup_logging(self):
        """
        Configures the logging system by adding specific handlers and formatters.
        This is preferred over logging.basicConfig in modular applications.
        """
        logging_str = "[%(asctime)s: %(levelname)s: %(module)s.%(funcName)s]: %(message)s"
        log_filepath = os.path.join(self._log_dir, self._log_file_name)
        os.makedirs(self._log_dir, exist_ok=True)

        # 1. Define Formatter
        formatter = logging.Formatter(logging_str, datefmt='%Y-%m-%d %H:%M:%S')

        # 2. File Handler (Writes logs to a file)
        file_handler = logging.FileHandler(log_filepath, mode='a')
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        # 3. Stream Handler (Writes logs to console/stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)
        
        self._logger.info("Logging initialized successfully.")
        
    @property
    def logger(self):
        """
        Provides access to the configured logger instance.
        """
        return self._logger

logger = Logger().logger