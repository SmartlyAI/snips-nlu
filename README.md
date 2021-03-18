
======================================================
Snips REST API service for NLU

# [POST] /v0.1/train

```json

inputs:
    { 
        'model': '3',
        'oldModel': '0',
        'project': '6040edd76134e10779d30ee6',
        'intents':
            [ { 'name': 'intentslug1',
                'elements': [ "comment dois je t'appeler", 'quel est ton nom' ] },
            { 'name': 'intentslug2',
                'elements':
                [ '@entity_6040eee6c43c7bf5a2b881c2:entity phrase', 'test here' ] } ],
        'slots':
            [ { 'id': '6040eee6c43c7bf5a2b881c2',
                'name': 'entity',
                'elements':
                [ { 'value': 'one', 'synonyms': [ 'syno', 'test syno' ] },
                { 'value': 'two', 'synonyms': [] } ] } ],
        'language': 'fr',
        'country': 'fr',
        'alphabet': null 
    }

outputs:
    { 
        'type': 'TrainStartedMessage' 
    }
```

# [GET] /v0.1/train/?project=dddddd&model=

```json
inputs:
 	{
        'project': '6035325d1d9c4316ac7d76c8'
    }
output:
     { 
       'type': 'TrainStatusResultMessage',
       'project': '6035325d1d9c4316ac7d76c8',
       'training': false,
       'current_component_name': '',
       'current_component_number': 0,
       'number_of_components': 0 
     }
```

# [POST] /v0.1/parse

```json
inputs:
    {
        'project': 'Id du graph', // en as t on vraiment besoin?
        'model': '6035325d1d9c4316ac7d76c8',
        'text': 'input de l'utilisateur'
    }

outputs:
    { 
        'type': 'ResultMessage',
        'intent':
        { 
            'name': 'intent 1',
            'confidence': 0.7041624784469604 
        },
        alternatives:
        [
            { 
                'name': 'intent 2',
                'confidence': 0.342 
            },
            { 
                'name': 'intent 3',
                'confidence': 0.31 
            },
            { 
                'name': 'intent 4',
                'confidence': 0.08 
            },
            { 
                'name': 'intent 5',
                'confidence': 0.15 
            },
       ],
       'entities': [
            'system_entites': {
                'number': [
                    {
                        'value': 11,
                        'unit': null,
                        'string': "11"
                    },
                ],
                'amount_of_money': [
                    {
                        'value': 50,
                        'unit': 'EUR',
                        'string': '50 euros'
                    }
                ],
            },   
        	'custom_entities': {
        		'jeuxVidéO': [
        			{
        				'value': 'jeux',
        				'string': 'jeux',
        				'learned': false
        			},
                ],
            },

        'sentiment':
            { 
                'label': 'neutral',
                'score': 0,
                'additional_informations': 
                { 
                    'negative': 0,
                    'neutral': 1,
                    'positive': 0,
                    'compound': 0 
                }
            },
        'text': 'quel est ton nom',
        'project': '6035325d1d9c4316ac7d76c8',
        'model': '4' 
    }
```

======================================================