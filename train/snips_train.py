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


from snips_nlu import SnipsNLUEngine
from datetime import datetime
from snips_nlu.dataset import Dataset
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from random import sample
import re
import yaml
import io
import shutil


__all__ = ['check_train_request_data', 'TrainSnipsNlu', 'get_time_server']


NFS_SERVER = Path(__file__).parent.parent / "nfs_server"
re_ents = re.compile(r'(@\w+_\w+:\w+)', re.I)

# python3 -m snips_nlu download-all-languages
# python3 -m snips_nlu download-language-entities en
# docker run --name redis -p 6379:6379 -d redis


def get_time_server():
    return datetime.now()

def get_human_size(folder):
    size = sum(fil.stat().st_size for fil in folder.rglob('*'))
    B = "B"
    KB = "KB" 
    MB = "MB"
    GB = "GB"
    TB = "TB"
    UNITS = [B, KB, MB, GB, TB]
    HUMANFMT = "%f %s"
    HUMANRADIX = 1024.
    for u in UNITS[:-1]:
        if size < HUMANRADIX : return HUMANFMT % (size, u)
        size /= HUMANRADIX
    return HUMANFMT % (size,  UNITS[-1])


def check_train_request_data(data):
    """[summary]

    Args:
        data ([type]): [description]

    Returns:
        [type]: [description]
    """
    for keys in ["oldModel","model","project","slots","language","country","alphabet"]:
        if keys not in data: return "{} not found".format(keys)
        elif type(data['intents']) != list or type(data['slots']) != list: 
            return "Type for {} key is not allowed".format(keys)
    return data


class TrainSnipsNlu():
    def __init__(self, request_data):
        self.request_data = request_data
        self.all_entities = {}
        self.smartly_entities = defaultdict(list)

    def build_snips_data_format(self):
        intents_requests  = self.request_data['intents']
        entities_requests = self.request_data['slots']
        len_utterances, len_intents, len_entities  = 0, 0, 0

        # Format entities from request data
        for r in entities_requests:
            for v in r['elements']:
                self.smartly_entities[r['id']].append((v['value'], v['synonyms']))

        # Format intents from request data
        snips_results_intents_data, snips_ents_data = [], []
        for content in intents_requests:
            intent_p, questions_p = content['name'], content['elements']
            s_intent = {}
            s_intent['type'] = 'intent'
            s_intent['name'] = intent_p
            len_intents += 1
            new_questions, new_entities = self.check_smartly_entity(questions_p, "snips")
            new_questions = list(set(new_questions)) 
            len_utterances += len(new_questions)
            s_intent['utterances'] = new_questions
            if len(new_entities.keys()) >= 1:
                ents = []
                for k, v in new_entities.items():
                    ents.append({"name": k, "entity": k})
                s_intent['slots'] = ents
            else: s_intent['slots'] = []
            
            snips_results_intents_data.append(s_intent)
            snips_ents_data.append(new_entities)
        
        f_entities = {}
        for ents in snips_ents_data:
            for k, v in ents.items():
                ent_snips = {}
                ent_data = self.smartly_entities[k]
                values_ents = [x[0] for x in ent_data]
                syn_ents = [x[1] for x in ent_data if x[1] != ['Synonyms']]
                all_values = values_ents + syn_ents[0]
                ent_snips['type'] = 'entity'
                ent_snips['name'] = k
                ent_snips['automatically_extensible'] = True
                ent_snips['matching_strictness'] = 0.8
                ent_snips['use_synonyms'] = True
                norm_values = []
                for x in list(set(all_values)):
                    if type(x) == str:
                        if "," in x: 
                            vf = [c.strip() for c in x.split(',') if c != '']
                            if len(vf) <= 1: norm_values.append(vf[0])
                            else: norm_values.append(vf)
                        else: norm_values.append(x)
                ent_snips['values'] = norm_values
                #print(ent_snips)
                f_entities[k] = ent_snips

        for k, v in f_entities.items(): 
            snips_results_intents_data.append(v)
        
        len_entities = len(f_entities.keys())

        # Build yaml file with data
        yaml_train_data = yaml.dump_all(snips_results_intents_data, 
                                        indent=2, allow_unicode=True, 
                                        line_break=True)
        DATASET_JSON = Dataset.from_yaml_files(self.request_data['language'], 
                                            [io.StringIO(yaml_train_data)]).json

        train_stats = {"len_utterances":len_utterances, 
                       "len_intents":len_intents, "len_entities":len_entities}
        return DATASET_JSON, train_stats

    def check_smartly_entity(self, questions, form="rasa"):
        for entity, ref in self.smartly_entities.items():
            #print('---- ', entity)
            val_ents = [x[0] for x in ref]
            syno_ents = self.split_syno([x[1] for x in ref])
            #if form == "snips": all_entities[entity] = val_ents
            #elif form == "rasa":
            all_test_ents = list(set(syno_ents + val_ents))
            self.all_entities[entity] = all_test_ents

        new_questions, entities = [], {}
        for question in questions:
            quest_repl = question
            search_data = re_ents.findall(question)
            if search_data:
                #print(question, ' -> \n', search_data)
                for all_match in search_data:
                    entity = all_match.split('_')[-1].split(':')[0]
                    #print('\t', all_match, ' | ', entity)
                    #print(self.all_entities)
                    try:
                        ref_value = self.all_entities[entity]
                        ent_rand = sample(ref_value, 1)[0]
                        #print(ent_rand)
                        entity_val = entity
                        if form == "snips":
                            #reg = '@.+_.+:{}'.format(entity)
                            quest_repl = re.sub(all_match, '[{}]({})'.format(entity_val, ent_rand), quest_repl)
                        elif form == "rasa":
                            quest_repl = re.sub(all_match, '[{}]({})'.format(ent_rand, entity_val), quest_repl)
                        entities[entity] = ref_value
                    except Exception as e: 
                        Exception('ERROR: {} - {} - {}'.format(str(e), entity, quest_repl))

                #print(orig, ' | ', quest_repl)
                new_questions.append(quest_repl)
                #print('\n----------------------------------\n')

            else: new_questions.append(question)

        return new_questions, entities

    def split_syno(self, syn):
        out = []
        for x in syn:
            if type(x) == str:
                for e in x.split(', '): out.append(e.replace(',', ''))
        return out

    def get_language_resource(self, language):
        """get language resource

        Args:
            language (string): user language

        Returns:
            dict: language resource configuration
        """
        if language == "fr":
            from snips_nlu.default_configs import CONFIG_FR
            return CONFIG_FR
        elif language == "en":
            from snips_nlu.default_configs import CONFIG_EN
            return CONFIG_EN
        elif language == "de":
            from snips_nlu.default_configs import CONFIG_DE
            return CONFIG_DE
        else: return Exception('Not available resource for this language')

    def build_train_instance(self, CONFIG, train_dataset, train_stats):
        try:               
            seed = 42
            engine = SnipsNLUEngine(config=CONFIG, random_state=seed)

            start = datetime.now()
            engine.fit(train_dataset)
            training_time = datetime.now() - start
            #print("Training time for {}: {}s".format(language, training_time))
            
            project_id = self.request_data['project']
            time_save = str(get_time_server().time()).replace(' ', '_')
            model_id = self.request_data['model']  
            temp_name = f'SnipsModel-{project_id}-{model_id}'
            model_save = NFS_SERVER / temp_name
            model_output_dir = Path(model_save)
            if model_output_dir.exists(): shutil.rmtree(model_output_dir)

            start = datetime.now()
            engine.persist(model_save)
            persist_time = datetime.now() - start

            model_size = get_human_size(model_output_dir)
            #print("Saving model time for {}: {}".format(language, persist_time))
            #print("Model size for {}: {}".format(language, model_size))
            
            stats_train_file = f"stats-{project_id}-{time_save}.yaml"
            stats_train_file = model_save / stats_train_file

            output_stats = {
                "persist_time": str(persist_time), "training_time": str(training_time),
                "model_size": model_size, "language": self.request_data['language'] , 
                "training_data": train_stats,
                "training_seed": seed, "project": project_id, 
                "model": self.request_data['model'], 
                "oldModel": self.request_data['oldModel'],
                "type": 'TrainStatusResultMessage',
                "training": False,
                "current_component_name": '',
                "current_component_number": 0,
                "number_of_components": 0 
            }
            with open(stats_train_file, "w") as stats_stream:
                yaml.dump(output_stats, stats_stream, indent=2, allow_unicode=True)

            return output_stats
        except Exception as e: 
            return {'message': f'Something happen during training data: {str(e)}',
                    'status': 'error'}, 400

    def train(self):
        return self.build_train_instance
