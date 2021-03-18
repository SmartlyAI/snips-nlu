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


from flask          import Flask, request, jsonify
from flask_restful  import Resource, Api
from flask_cors     import CORS
from flask_executor import Executor
from flask_sse      import sse

from .snips_train import *
from .utils import *

smartly_train_app = Flask(__name__)
smartly_train_app.config.from_object(ApiConfigDev)
smartly_train_app.app_context().push()


executor = Executor(smartly_train_app)
CORS(smartly_train_app, resources={r"/*": {"origin": "*"}})
smartly_train_api = Api(smartly_train_app, catch_all_404s=True, 
                        serve_challenge_on_401=True, 
                        default_mediatype='json')


def training_callback(future):
    smartly_train_app.app_context().push()
    """
    'add_done_callback', 'cancel', 'cancelled', 'done', 'exception', 
    'result', 'running', 
    'set_exception', 'set_result', 'set_running_or_notify_cancel'
    print(future.done(), future.result(), future.cancelled(), 
            future.exception(), future.running())
    """
    #future.set_running_or_notify_cancel()
    if future.running():
        sse.publish(future.result(), type='snips_training_running')
    if future.done():
        sse.publish(future.result(), type='snips_training_done')
        return jsonify(future.result()), 200
    if future.cancelled():
        sse.publish(future.result(), type='snips_training_cancel')    
        return jsonify(future.result()), 400
    if future.exception():
        sse.publish(future.result(), type='snips_training_error')
        return jsonify(future.result()), 400



class SmartlySnipsTrainResource(Resource):
    def get(self):
        params = request.args
        project_id = params.get('project')
        model_version = params.get('model')
        executor_id = project_id+"-"+model_version
        task_ok = executor.futures.done(executor_id)
        print(task_ok)
        if task_ok == None:
            return  {'type': 'TrainStatusResultMessage',
                    'project': params['project'],
                    'training': False,
                    'current_component_name': '',
                    'current_component_number': 0,
                    'number_of_components': 0 }, 400

        if not task_ok:
            state = executor.futures._state(executor_id)
            return {'status': state,
                    'type': 'TrainStatusResultMessage',
                    'project': params['project'],
                    'training': not task_ok,
                    'current_component_name': '',
                    'current_component_number': 0,
                    'number_of_components': 0 }, 200 if state == "RUNNING" else 400


        future = executor.futures.pop(executor_id)
        executor.futures.add(executor_id, future)
        result_data = future.result()
        return {'status': 'done', 
                **result_data,
                'type': 'TrainStatusResultMessage',
                'project': params['project'],
                'training': False,
                'current_component_name': '',
                'current_component_number': 0,
                'number_of_components': 0 }, 200


    def post(self):
        request_data = check_train_request_data(request.json)
        if type(request_data) != dict: 
            return {"message": request_data, "status": "error"}, 400

        project  = request_data['project']
        
        executor_id = project+'-'+request_data['model']
        if executor.futures.done(executor_id):
            future = executor.futures.pop(executor_id)
            executor.futures.add(executor_id, future)
            return {"message": "We already train this model: executor id exist", 
                    "executor_id": executor_id, "results": future.result()}, 409

        train_engine = TrainSnipsNlu(request_data)

        try:
            train_dataset, train_stats = train_engine.build_snips_data_format()  
        except Exception as e: 
            return {'message': f'Something happen during building training data: {str(e)}',
                    'status': 'error'}, 400
        try:
            language = request_data['language']  
            CONFIG = train_engine.get_language_resource(language)
        except Exception as e: 
            return {'message': f'Something happen during building snips language resource: {str(e)}',
                    'status': 'error'}, 400

        function_executor = train_engine.train()

        try: executor.submit_stored(executor_id, function_executor, 
                                CONFIG, train_dataset, train_stats)
        except Exception as e: 
            return {'message': f'Something happen during training data: {str(e)}',
                    'status': 'error'}, 400

        return {"type": "TrainStartedMessage", 
                "executor_id": executor_id}, 200




smartly_train_api.add_resource(SmartlySnipsTrainResource, '/')
smartly_train_app.register_blueprint(sse, url_prefix='/stream')
executor.add_default_done_callback(training_callback)


__all__ = ['smartly_train_app']