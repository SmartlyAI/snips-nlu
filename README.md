
======================================================
Getting a result from a trained bot
url: /parse
method: POST

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
======================================================
Asking a bot to be trained
url: /train:
method: POST

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

======================================================
Asking for the training status of a model
method: GET

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
======================================================




Snips
======

[
  {
    "rawValue": "Paris",
    "value": {
      "kind": "Custom",
      "value": "Paris"
    },
    "entity": "location",
    "slotName": "departure",
    "range": {
      "start": 28,
      "end": 41
    }
  },
  {
    "rawValue": "Tokyo",
    "value": {
      "kind": "Custom",
      "value": "Tokyo"
    },
    "entity": "location",
    "slotName": "arrival",
    "range": {
      "start": 28,
      "end": 41
    }
  }
]

Smartly.ai
===========

'custom_entities': {
    'location': [
        {
            'value': 'Paris',
            'string': 'Paris',
            'learned': true
        },
        {
            'value': 'Tokyo',
            'string': 'Tokyo',
            'learned': true
        },       
    ],
},


====================================



TIME
// time: today, Monday, Feb 18, the 1st of march

Smartly/time

"time": 
[
{
  "value": "2021-03-16T00:00:00.000+01:00",
  "unit": null,
  "string": "aujourd hui"
}

Snips/datetime

[
  {
    "kind": "InstantTime",
    "value": "2017-06-13 18:00:00 +02:00",
    "grain": "Hour",
    "precision": "Exact"
  }
]

[
  {
    "kind": "InstantTime",
    "value": "2017-06-13 00:00:00 +02:00",
    "grain": "Day",
    "precision": "Exact"
  }
]





TEMPERATURE
// temperature: 70°, 72° Fahrenheit, thirty two celsius

Smartly/temperature
"temperature": [
  {
    "value": 20,
    "unit": "degree",
    "string": "20°"
  }

Snips/temperature
[
  {
    "kind": "Temperature",
    "value": 23.0,
    "unit": "celsius"
  },
  {
    "kind": "Temperature",
    "value": 60.0,
    "unit": "fahrenheit"
  }
]



NUMBER
// number: eighteen, 0.77, 100K

Smartly/number
"number": [
  {
    "value": 11,
    "unit": null,
    "string": "11"
  }
]

Snips/number
[
  {
    "kind": "Number",
    "value": 42.0
  }
]



ORDINAL
// ordinal: 4th, first, seventh

Smartly/ordinal
"ordinal": [
  {
    "value": 2,
    "unit": null,
    "string": "second"
  }
],

Snips/ordinal
[
  {
    "kind": "Ordinal",
    "value": 2
  }
]



AMOUNT OF MONEY
// amount of money: ten dollars, 4 bucks, $20

Smartly/amount_of_money
"amount_of_money": [
  {
    "value": 50,
    "unit": "EUR",
    "string": "50 euros"
  }
],

Snips
[
  {
    "kind": "AmountOfMoney",
    "value": 10.05,
    "precision": "Approximate",
    "unit": "€"
  }
]

DURATION
// duration: 2 hours,  4 days,  3 minutes
Smartly/duration
"duration": [
  {
    "value": 3,
    "unit": "month",
    "string": "3 mois"
  }
],

Snips
[
  {
    "kind": "Duration",
    "years": 0,
    "quarters": 0,
    "months": 3,
    "weeks": 0,
    "days": 0,
    "hours": 0,
    "minutes": 0,
    "seconds": 0,
    "precision": "Exact"
  }
]




snips-nlu-city-de @ https://resources.snips.ai/nlu/gazetteer-entities/de/city/v0.2/snips_nlu_city_de-0.2.0.tar.gz
snips-nlu-city-en @ https://resources.snips.ai/nlu/gazetteer-entities/en/city/v0.2/snips_nlu_city_en-0.2.0.tar.gz
snips-nlu-city-es @ https://resources.snips.ai/nlu/gazetteer-entities/es/city/v0.2/snips_nlu_city_es-0.2.0.tar.gz
snips-nlu-city-fr @ https://resources.snips.ai/nlu/gazetteer-entities/fr/city/v0.2/snips_nlu_city_fr-0.2.0.tar.gz
snips-nlu-country-de @ https://resources.snips.ai/nlu/gazetteer-entities/de/country/v0.2/snips_nlu_country_de-0.2.0.tar.gz
snips-nlu-country-en @ https://resources.snips.ai/nlu/gazetteer-entities/en/country/v0.2/snips_nlu_country_en-0.2.0.tar.gz
snips-nlu-country-es @ https://resources.snips.ai/nlu/gazetteer-entities/es/country/v0.2/snips_nlu_country_es-0.2.0.tar.gz
snips-nlu-country-fr @ https://resources.snips.ai/nlu/gazetteer-entities/fr/country/v0.2/snips_nlu_country_fr-0.2.0.tar.gz
snips-nlu-de @ https://github.com/snipsco/snips-nlu-language-resources/releases/download/snips_nlu_de-0.2.3/snips_nlu_de-0.2.3.tar.gz
snips-nlu-en @ https://github.com/snipsco/snips-nlu-language-resources/releases/download/snips_nlu_en-0.2.3/snips_nlu_en-0.2.3.tar.gz
snips-nlu-es @ https://github.com/snipsco/snips-nlu-language-resources/releases/download/snips_nlu_es-0.2.2/snips_nlu_es-0.2.2.tar.gz
snips-nlu-fr @ https://github.com/snipsco/snips-nlu-language-resources/releases/download/snips_nlu_fr-0.2.4/snips_nlu_fr-0.2.4.tar.gz
snips-nlu-it @ https://github.com/snipsco/snips-nlu-language-resources/releases/download/snips_nlu_it-0.1.2/snips_nlu_it-0.1.2.tar.gz
snips-nlu-ja @ https://github.com/snipsco/snips-nlu-language-resources/releases/download/snips_nlu_ja-0.2.1/snips_nlu_ja-0.2.1.tar.gz
snips-nlu-ko @ https://github.com/snipsco/snips-nlu-language-resources/releases/download/snips_nlu_ko-0.2.0/snips_nlu_ko-0.2.0.tar.gz
snips-nlu-musicalbum-de @ https://resources.snips.ai/nlu/gazetteer-entities/de/musicalbum/v0.2/snips_nlu_musicalbum_de-0.2.0.tar.gz
snips-nlu-musicalbum-en @ https://resources.snips.ai/nlu/gazetteer-entities/en/musicalbum/v0.2/snips_nlu_musicalbum_en-0.2.5.tar.gz
snips-nlu-musicalbum-es @ https://resources.snips.ai/nlu/gazetteer-entities/es/musicalbum/v0.2/snips_nlu_musicalbum_es-0.2.0.tar.gz
snips-nlu-musicalbum-fr @ https://resources.snips.ai/nlu/gazetteer-entities/fr/musicalbum/v0.2/snips_nlu_musicalbum_fr-0.2.1.tar.gz
snips-nlu-musicartist-de @ https://resources.snips.ai/nlu/gazetteer-entities/de/musicartist/v0.2/snips_nlu_musicartist_de-0.2.0.tar.gz
snips-nlu-musicartist-en @ https://resources.snips.ai/nlu/gazetteer-entities/en/musicartist/v0.2/snips_nlu_musicartist_en-0.2.3.tar.gz
snips-nlu-musicartist-es @ https://resources.snips.ai/nlu/gazetteer-entities/es/musicartist/v0.2/snips_nlu_musicartist_es-0.2.0.tar.gz
snips-nlu-musicartist-fr @ https://resources.snips.ai/nlu/gazetteer-entities/fr/musicartist/v0.2/snips_nlu_musicartist_fr-0.2.1.tar.gz
snips-nlu-musictrack-de @ https://resources.snips.ai/nlu/gazetteer-entities/de/musictrack/v0.2/snips_nlu_musictrack_de-0.2.0.tar.gz
snips-nlu-musictrack-en @ https://resources.snips.ai/nlu/gazetteer-entities/en/musictrack/v0.2/snips_nlu_musictrack_en-0.2.5.tar.gz
snips-nlu-musictrack-es @ https://resources.snips.ai/nlu/gazetteer-entities/es/musictrack/v0.2/snips_nlu_musictrack_es-0.2.0.tar.gz
snips-nlu-musictrack-fr @ https://resources.snips.ai/nlu/gazetteer-entities/fr/musictrack/v0.2/snips_nlu_musictrack_fr-0.2.1.tar.gz
snips-nlu-parsers==0.4.3
snips-nlu-pt-br @ https://github.com/snipsco/snips-nlu-language-resources/releases/download/snips_nlu_pt_br-0.1.1/snips_nlu_pt_br-0.1.1.tar.gz
snips-nlu-pt-pt @ https://github.com/snipsco/snips-nlu-language-resources/releases/download/snips_nlu_pt_pt-0.1.1/snips_nlu_pt_pt-0.1.1.tar.gz
snips-nlu-region-de @ https://resources.snips.ai/nlu/gazetteer-entities/de/region/v0.2/snips_nlu_region_de-0.2.0.tar.gz
snips-nlu-region-en @ https://resources.snips.ai/nlu/gazetteer-entities/en/region/v0.2/snips_nlu_region_en-0.2.0.tar.gz
snips-nlu-region-es @ https://resources.snips.ai/nlu/gazetteer-entities/es/region/v0.2/snips_nlu_region_es-0.2.0.tar.gz
snips-nlu-region-fr @ https://resources.snips.ai/nlu/gazetteer-entities/fr/region/v0.2/snips_nlu_region_fr-0.2.0.tar.gz


