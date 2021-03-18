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


from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_executor import Executor

from train.snips_train import *

smartly_home_app = Flask(__name__)

executor = Executor(smartly_home_app)
smartly_home_api = Api(smartly_home_app)


__all__ = ['smartly_home_app']


class SmartlySnipsTrainStatusResource(Resource):
    def get(self, status_id):
        if not executor.futures.done(status_id):
            return {'status': executor.futures._state(status_id)}, 200

        future = executor.futures.pop(status_id)
        return {'status': done, 'result': future.result()}, 200



smartly_home_api.add_resource(SmartlySnipsTrainStatusResource, '/<status_id>')
