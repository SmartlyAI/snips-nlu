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


from snips_nlu import SnipsNLUEngine
from datetime import datetime
from pathlib import Path
from os.path import getctime
from collections import defaultdict


__all__ = ['check_parse_request_data', 'ParseSnipsNlu']


NFS_SERVER = Path(__file__).parent.parent / "nfs_server"

SNIPS_ENGINE_MEMORY = {}


def check_parse_request_data(data):
    """[summary]

    Args:
        data ([type]): [description]

    Returns:
        [type]: [description]
    """
    for keys in ["project","model","text"]:
        if keys not in data: return "{} not found".format(keys)
        elif type(data[keys]) != str: return "Type for {} key is not allowed".format(keys)
    return data


def get_model_path(project_id, model_version):
    project = list(NFS_SERVER.glob('*{}-{}'.format(project_id, model_version)))
    sort_project = sorted(project, key=getctime, reverse=True)
    if sort_project: return project[0]
    else: 
        return Exception(f'Not model path found for this: {project_id}')


class ParseSnipsNlu():
    def __init__(self, request_data):
        global SNIPS_ENGINE_MEMORY
        self.request_data = request_data
        self.model_version = request_data['model']
        self.project_id = request_data['project']
        self.model_id = self.project_id+'-'+self.model_version
        self.model_path = get_model_path(self.project_id, self.model_version)
        if self.model_id in SNIPS_ENGINE_MEMORY:
            self.engine = SNIPS_ENGINE_MEMORY[self.model_id]
        else: 
            self.engine = SnipsNLUEngine.from_path(self.model_path)
            SNIPS_ENGINE_MEMORY[self.model_id] = self.engine

    def parse_engine(self, input_text, top=3):
        try: self.input_parse = self.engine.parse(input_text, top_n=top)
        except Exception as e: 
            return {'message': f'Something happen during parsing data: {str(e)}',
                    'status': 'error'}

    def parse_format(self):
        results = {}
        results['type'] = 'ResultMessage'
        best_model_result = self.input_parse[0]
        if best_model_result:
            results['intent'] = {
                "name": best_model_result['intent']['intentName'],
                "confidence": best_model_result['intent']['probability']
            }
            results['alternatives'] = self.format_alternatives(self.input_parse[1:2])
            results['entities'] = self.format_entities(best_model_result['slots'])
            results['text'] = self.request_data['text']
            results['project'] = self.project_id
            results['model'] = self.model_version
            return results
        else: Exception("Can't classify data for this results")

    def format_entities(self, entities):
        results, customs, systems = {}, defaultdict(list), defaultdict(list)
        for slot in entities:
            if slot['value']['kind'] == 'Custom':
                customs[slot['entity']].append(
                    {
                        'value': slot['value']['value'],
                        'string': slot['rawValue'],
                        'learned': True
                    })
                
            elif slot['value']['kind'] == 'Custom':
                systems[slot['entity']].append(
                    {
                        'value': slot['value']['value'],
                        'string': slot['rawValue'],
                        'learned': True
                    }
                )
        results['system_entites'] = {x:v for x, v in systems.items()}
        results['custom_entities'] = {x:v for x, v in customs.items()}
        return results

    def format_alternatives(self, entities):
        return [ {"name":x['intent']['intentName'],
                  "confidence":x['intent']['probability'] } 
                for x in entities]
        
        



