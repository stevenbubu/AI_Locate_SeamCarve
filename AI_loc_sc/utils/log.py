import logging.config
'''
Level       Numeric value
CRITICAL    50
ERROR       40
WARNING     30
INFO        20
DEBUG       10
NOTSET      0
'''

config = {
    'version': 1,
    'formatters': {
        'simple': {
            'format': '%(asctime)s - %(filename)s - %(lineno)s - %(module)s - %(levelname)s - %(message)s',
        },
        # 其他的 formatter
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple'
        },
        # 'file': {
        #     'class': 'logging.FileHandler',
        #     'filename': 'logging.log',
        #     'level': 'DEBUG',
        #     'formatter': 'simple'
        # },
        # 其他的 handler
    },
    'loggers':{
        'StreamLogger': {
            'handlers': ['console'],
            'level': 'INFO',
        },
        # 'FileLogger': {
        #     # 既有 console Handler，还有 file Handler
        #     'handlers': ['console', 'file'],
        #     'level': 'DEBUG',
        # },
        # 其他的 Logger
    }
}

# logging.config.dictConfig(config)
# logger = logging.getLogger("StreamLogger")
# FileLogger = logging.getLogger("FileLogger")

# logger.debug('debug message')
# logger.info('info message')
# logger.warn('warn message')
# logger.error('error message')
# logger.critical('critical message')

