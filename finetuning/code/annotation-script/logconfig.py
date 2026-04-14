from logging import Logger, StreamHandler, Formatter, getLogger, INFO


LOG_FILE_NAME_FORMAT = "{:%Y-%m-%d}"
LOG_MESSAGE_FORMAT = "%(asctime)s %(levelname)s --- [%(module)-12s] : %(message)-12s"

def create_logger() -> Logger:
    stream_handler = StreamHandler()
    formatter = Formatter(LOG_MESSAGE_FORMAT)
    
    stream_handler.setFormatter(formatter)
    
    logger = getLogger(__name__)
    logger.addHandler(stream_handler)
    logger.setLevel(INFO)

    return logger