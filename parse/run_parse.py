#!/usr/bin/python3
# coding: utf-8 

from __future__ import unicode_literals

"""
    Smartly - Digital et Data :  Snips Parse (REST API)
    ------------------------------------------------------
    
    This module test all functionalities of Snips API.
    
    :copyright: © 2020 by Elvis.
    :license: Creative Commons, see LICENSE for more details.
"""


from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors    import CORS

from .snips_parse import *


smartly_parse_app = Flask(__name__)

CORS(smartly_parse_app, resources={r"/*": {"origins": "*"}})
smartly_parse_api = Api(smartly_parse_app,
                        catch_all_404s=True, 
                        serve_challenge_on_401=True, 
                        default_mediatype='json')


class SmartlySnipsParseResource(Resource):
    def get(self):
        return {'hello': 'SmartlySnipsParseResource'}
    
    def post(self):
        request_data = check_parse_request_data(request.json)
        project  = request_data['project']
        model_id = request_data['model']
        user_input = request_data['text']
        
        engine = ParseSnipsNlu(request_data)
        parse_data = engine.parse_engine(user_input)

        result_data = engine.parse_format()

        return result_data, 200




smartly_parse_api.add_resource(SmartlySnipsParseResource, '/')


__all__ = ['smartly_parse_app']