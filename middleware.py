#!/usr/bin/python3
# coding: utf-8 

from __future__ import unicode_literals

"""
    Smartly - Digital et Data : Snips Train Service (REST API)
    ------------------------------------------------------
    
    This module test all functionalities of Snips API.
    
    :author: MTE
    :copyright: © 2020 by Smartly and OBS D&D
    :license: Smartly, all rights reserved
"""

__version__ = "0.1"


from flask import Flask
from flask_cors     import CORS
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from parse.run_parse import smartly_parse_app as parse_service
from train.run_train import smartly_train_app as train_service
from home import smartly_home_app as home_service

# Build endpoint with version number
parse_endpoint = f"/v{__version__}/parse"
train_endpoint = f"/v{__version__}/train"

# Build Middleware for services
snips_middleware = DispatcherMiddleware(home_service, {
    parse_endpoint: parse_service,
    train_endpoint: train_service
})

snips_app = Flask(__name__)
snips_app.app_context().push()
CORS(snips_app, resources={r"/*": {"origins": "*"}})
snips_app.wsgi_app = snips_middleware


if __name__ == '__main__':
    from os import environ
    snips_app.run(
        host=environ['SNIPS_HOST'],
        port=environ['SNIPS_PORT'],
        use_evalex=True,
        use_reloader=environ['SNIPS_RELOADER'],
        use_debugger=environ['SNIPS_DEBUG']
    )