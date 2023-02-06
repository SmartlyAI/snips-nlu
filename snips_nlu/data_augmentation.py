from __future__ import unicode_literals

from builtins import next
from copy import deepcopy
from itertools import cycle

from future.utils import iteritems

from snips_nlu.constants import (
    CAPITALIZE, DATA, ENTITIES, ENTITY, INTENTS, TEXT, UTTERANCES, ARABE_VARIANTS)
from snips_nlu.entity_parser.builtin_entity_parser import is_builtin_entity
from snips_nlu.languages import get_default_sep
from snips_nlu.preprocessing import tokenize_light
from snips_nlu.resources import get_stop_words


def capitalize(text, language, resources):
    tokens = tokenize_light(text, language)
    stop_words = get_stop_words(resources)
    return get_default_sep(language).join(
        t.title() if t.lower() not in stop_words
        else t.lower() for t in tokens)


def capitalize_utterances(utterances, entities, language, ratio, resources,
                          random_state):
    capitalized_utterances = []
    for utterance in utterances:
        capitalized_utterance = deepcopy(utterance)
        for i, chunk in enumerate(capitalized_utterance[DATA]):
            capitalized_utterance[DATA][i][TEXT] = chunk[TEXT].lower()
            if ENTITY not in chunk:
                continue
            entity_label = chunk[ENTITY]
            if is_builtin_entity(entity_label):
                continue
            if not entities[entity_label][CAPITALIZE]:
                continue
            if random_state.rand() > ratio:
                continue
            capitalized_utterance[DATA][i][TEXT] = capitalize(
                chunk[TEXT], language, resources)
        capitalized_utterances.append(capitalized_utterance)
    return capitalized_utterances


def generate_utterance(contexts_iterator, entities_iterators):

    # Intent circular list that holds data dictionary:
    context = deepcopy(next(contexts_iterator))

    # List to fill with final augmented data:
    context_data = []

    # For each 'text'-utt dict (i.e. chunk):
    for chunk in context[DATA]:

        # If the "entity" key is in the dictionary (i.e. if there's an entity in the utterance):
        if ENTITY in chunk:

            # Get first entity in utterance:
            chunk[TEXT] = deepcopy(next(entities_iterators[chunk[ENTITY]]))
        
        # Append empty space to utterance:
        chunk[TEXT] = chunk[TEXT].strip() + " "

        # Append the final reseult:
        context_data.append(chunk)
    
    # Over-write the old dictionary entry:
    context[DATA] = context_data

    return context


def get_contexts_iterator(dataset, intent_name, random_state):
    shuffled_utterances = random_state.permutation(
        dataset[INTENTS][intent_name][UTTERANCES])
    return cycle(shuffled_utterances)


def get_entities_iterators(intent_entities, language, add_builtin_entities_examples, random_state):
    from snips_nlu_parsers import get_builtin_entity_examples

    # Empty dict:
    entities_its = dict()

    # For each entity:
    for entity_name, entity in iteritems(intent_entities):

        # Shuffled utterances:
        utterance_values = random_state.permutation(sorted(entity[UTTERANCES]))

        # Add builtin entity examples:
        if add_builtin_entities_examples and is_builtin_entity(entity_name):

            # Check if language is Arabic:
            if language in ARABE_VARIANTS:
                entity_examples = get_builtin_entity_examples(entity_name, 'fr')
            else: 
                entity_examples = get_builtin_entity_examples(entity_name, language)

            # Builtin entity examples must be kept first in the iterator to
            # ensure that they are used when augmenting data
            iterator_values = entity_examples + list(utterance_values)
        else:
            iterator_values = utterance_values
        
        # Create circular list of 'data' dictionaries:
        entities_its[entity_name] = cycle(iterator_values)
    return entities_its


def get_intent_entities(dataset, intent_name):
    intent_entities = set()
    for utterance in dataset[INTENTS][intent_name][UTTERANCES]:
        for chunk in utterance[DATA]:
            if ENTITY in chunk:
                intent_entities.add(chunk[ENTITY])
    return sorted(intent_entities)


def num_queries_to_generate(dataset, intent_name, min_utterances):
    nb_utterances = len(dataset[INTENTS][intent_name][UTTERANCES])
    return max(nb_utterances, min_utterances)


def augment_utterances(dataset, intent_name, language, min_utterances, capitalization_ratio, add_builtin_entities_examples, resources, random_state):

    # Returns circular list of utterances for the current intent:
    contexts_it = get_contexts_iterator(dataset, intent_name, random_state)

    # Dictionary of entities:
    intent_entities = {e: dataset[ENTITIES][e] for e in get_intent_entities(dataset, intent_name)}

    # Circular list for entities:
    entities_its = get_entities_iterators(intent_entities, language, add_builtin_entities_examples, random_state)

    # Integer for number of utterances to generate => maximum between "min_utterances" parameter and number of utterances in intent:
    nb_to_generate = num_queries_to_generate(dataset, intent_name, min_utterances)

    # Start generating utterances:
    generated_utterances = []

    # While nb_to_generate is still positive (not exhausted by decrementing to 0):
    while nb_to_generate > 0:
        generated_utterance = generate_utterance(contexts_it, entities_its)
        generated_utterances.append(generated_utterance)
        nb_to_generate -= 1

    # Returns capitalized version of the utterances:
    generated_utterances = capitalize_utterances(generated_utterances, dataset[ENTITIES], language, ratio=capitalization_ratio, resources=resources, random_state=random_state)

    return generated_utterances
