__all__ = ['ApiConfigDev', 'ApiConfigProd']

from os import environ

class ApiConfigDev(object):
    EXECUTOR_TYPE = "thread"
    EXECUTOR_MAX_WORKERS = 5
    EXECUTOR_PROPAGATE_EXCEPTIONS = True
    SSE_REDIS_URL = "redis://localhost:6379/0"
    SECRET_KEY = "smartly-api-snips-nlu-2021-digital-data"

class ApiConfigProd(ApiConfigDev):
    EXECUTOR_TYPE = "thread"
    EXECUTOR_MAX_WORKERS = 10
    EXECUTOR_PROPAGATE_EXCEPTIONS = True
    SSE_REDIS_URL = "redis://localhost:6379/0"
    SECRET_KEY = "smartly-api-snips-nlu-2021-digital-data"