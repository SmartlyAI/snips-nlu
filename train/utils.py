#!/usr/bin/python3
# coding: utf-8 

from __future__ import unicode_literals

"""
    Smartly - Digital et Data :  Snips Train (REST API)
    ------------------------------------------------------
    
    This module test all functionalities of Snips API.
    
    :copyright: © 2020 by Elvis.
    :license: Creative Commons, see LICENSE for more details.
"""

__version__ = "0.1"


__all__ = ['ApiConfigDev', 'ApiConfigProd']


class ApiConfigDev(object):
    EXECUTOR_TYPE = "thread"
    EXECUTOR_MAX_WORKERS = 5
    EXECUTOR_PROPAGATE_EXCEPTIONS = True
    SSE_REDIS_URL = "redis://127.0.0.1:6379"
    SECRET_KEY = "smartly-api-snips-nlu-2021-digital-data"


class ApiConfigProd(ApiConfigDev):
    EXECUTOR_TYPE = "thread"
    EXECUTOR_MAX_WORKERS = 10
    EXECUTOR_PROPAGATE_EXCEPTIONS = True
    SSE_REDIS_URL = "redis://localhost:6379/0"
    SECRET_KEY = "smartly-api-snips-nlu-2021-digital-data"