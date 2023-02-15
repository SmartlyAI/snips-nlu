from __future__ import division, unicode_literals

import itertools
import re
from builtins import next, range, str
from copy import deepcopy
from uuid import uuid4

from future.utils import iteritems, itervalues

from snips_nlu.constants import (DATA, ENTITY, INTENTS, TEXT,
                                 UNKNOWNWORD, UTTERANCES)
from snips_nlu.data_augmentation import augment_utterances
from snips_nlu.dataset import get_text_from_chunks
from snips_nlu.entity_parser.builtin_entity_parser import is_builtin_entity
from snips_nlu.preprocessing import tokenize_light
from snips_nlu.resources import get_noise

NOISE_NAME = str(uuid4())
WORD_REGEX = re.compile(r"\w+(\s+\w+)*")
UNKNOWNWORD_REGEX = re.compile(r"%s(\s+%s)*" % (UNKNOWNWORD, UNKNOWNWORD))


def remove_builtin_slots(dataset):
    filtered_dataset = deepcopy(dataset)
    for intent_data in itervalues(filtered_dataset[INTENTS]):
        for utterance in intent_data[UTTERANCES]:
            utterance[DATA] = [
                chunk for chunk in utterance[DATA]
                if ENTITY not in chunk or not is_builtin_entity(chunk[ENTITY])]
    return filtered_dataset


def get_regularization_factor(dataset):
    import numpy as np

    intents = dataset[INTENTS]
    nb_utterances = [len(intent[UTTERANCES]) for intent in itervalues(intents)]
    avg_utterances = np.mean(nb_utterances)
    total_utterances = sum(nb_utterances)
    alpha = 1.0 / (4 * (total_utterances + 5 * avg_utterances))
    return alpha


def get_noise_it(noise, mean_length, std_length, random_state):
    it = itertools.cycle(noise)
    while True:
        noise_length = int(random_state.normal(mean_length, std_length))
        # pylint: disable=stop-iteration-return
        yield " ".join(next(it) for _ in range(noise_length))
        # pylint: enable=stop-iteration-return


def generate_smart_noise(noise, augmented_utterances, replacement_string,
                         language):
    text_utterances = [get_text_from_chunks(u[DATA])
                       for u in augmented_utterances]
    vocab = [w for u in text_utterances for w in tokenize_light(u, language)]
    vocab = set(vocab)
    return [w if w in vocab else replacement_string for w in noise]


def generate_noise_utterances(augmented_utterances, noise, num_intents,
                              data_augmentation_config, language,
                              random_state):
    import numpy as np

    if not augmented_utterances or not num_intents:
        return []
    avg_num_utterances = len(augmented_utterances) / float(num_intents)
    if data_augmentation_config.unknown_words_replacement_string is not None:
        noise = generate_smart_noise(
            noise, augmented_utterances,
            data_augmentation_config.unknown_words_replacement_string,
            language)

    noise_size = min(
        int(data_augmentation_config.noise_factor * avg_num_utterances),
        len(noise))
    utterances_lengths = [
        len(tokenize_light(get_text_from_chunks(u[DATA]), language))
        for u in augmented_utterances]
    mean_utterances_length = np.mean(utterances_lengths)
    std_utterances_length = np.std(utterances_lengths)
    noise_it = get_noise_it(noise, mean_utterances_length,
                            std_utterances_length, random_state)
    # Remove duplicate 'unknownword unknownword'
    return [
        text_to_utterance(UNKNOWNWORD_REGEX.sub(UNKNOWNWORD, next(noise_it)))
        for _ in range(noise_size)]


def add_unknown_word_to_utterances(utterances, replacement_string,
                                   unknown_word_prob, max_unknown_words,
                                   random_state):
    if not max_unknown_words:
        return utterances


    new_utterances = deepcopy(utterances)
    for u in new_utterances:
        if random_state.rand() < unknown_word_prob:
            num_unknown = random_state.randint(1, max_unknown_words + 1)
            # We choose to put the noise at the end of the sentence and not
            # in the middle so that it doesn't impact to much ngrams
            # computation
            extra_chunk = {
                TEXT: " " + " ".join(
                    replacement_string for _ in range(num_unknown))
            }
            u[DATA].append(extra_chunk)
    return new_utterances


def build_training_data(dataset, language, data_augmentation_config, resources,random_state):

    import numpy as np

    # Create class mapping

    # Dictionary of intent IDs and their utterances, entities:
    intents = dataset[INTENTS]

    # Mapping dictionary of class name-index:
    intent_index = 0
    classes_mapping = dict()
    for intent in sorted(intents):
        classes_mapping[intent] = intent_index
        intent_index += 1

    # Noise class' index is the last index:
    noise_class = intent_index
    

    # Augment utterances:
    augmented_utterances = []
    utterance_classes = []

    # For each intent:
    for intent_name, intent_data in sorted(iteritems(intents)):

        # Number of utterances in current intent:
        nb_utterances = len(intent_data[UTTERANCES])

        # Minimum of utterances is the biggest of the number of utterances and "min_utterances" parameter:
        min_utterances_to_generate = max(data_augmentation_config.min_utterances, nb_utterances)

        # Original utterances + augmented ones:
        utterances = augment_utterances(
            dataset,
            intent_name,
            language=language,
            min_utterances=min_utterances_to_generate,
            capitalization_ratio = 0.0,
            add_builtin_entities_examples = data_augmentation_config.add_builtin_entities_examples,
            resources=resources,
            random_state=random_state)

        # Append final utterances:
        augmented_utterances += utterances

        # Create and append indices for the new augmented data:
        utterance_classes += [classes_mapping[intent_name] for _ in range(len(utterances))]


    # Skipped by default:
    if data_augmentation_config.unknown_words_replacement_string is not None:
        augmented_utterances = add_unknown_word_to_utterances(
            augmented_utterances,
            data_augmentation_config.unknown_words_replacement_string,
            data_augmentation_config.unknown_word_prob,
            data_augmentation_config.max_unknown_words,
            random_state
        )

    # Adding noise
    # List of noise utterances from noise.txt file:
    noise = get_noise(resources)

    # Dynamically remove dataset utterances from noise utterances:
    dataset_unique_words = set()

    # Build set of unique words in user's dataset:
    for utterance_dict in intents.values():
        for data_dict in utterance_dict['utterances']:
            for text in data_dict['data']:
                for word in text['text'].split():
                    dataset_unique_words.add(word)

    # Remove words from noise utterance if words exist on dataset:
    for i, noisy_utt in enumerate(noise):
        for word in noisy_utt.split():
            if word in dataset_unique_words:

                # If utt has only one word => remove whole utterance so we're not left with empty string:
                if len(noisy_utt.split()) == 1:
                    del noisy_utt[i]
                else:
                    noise[i] = noisy_utt.replace(word, '').strip()
    
    
    # Format the noise utterances into data dict like the other utterances:
    noisy_utterances = generate_noise_utterances(augmented_utterances, noise, len(intents), data_augmentation_config, language, random_state)

    # Add noisy utterances to final augmented utterances:
    augmented_utterances += noisy_utterances

    # Append every data dict of the noise class to the augmented classes:
    utterance_classes += [noise_class for _ in noisy_utterances]

    # If there are noisy utterances:
    if noisy_utterances:

        # Its index mapping will be the last int index "noise_class" arrived at:
        classes_mapping[NOISE_NAME] = noise_class

    # Number of classes:
    nb_classes = len(set(itervalues(classes_mapping)))

    # List of Nones:
    intent_mapping = [None for _ in range(nb_classes)]

    # For each intent name & intent index pair:
    for intent, intent_class in iteritems(classes_mapping):

        # If it's the None intent:
        if intent == NOISE_NAME:

            # Put it's index as key and None as its name:
            intent_mapping[intent_class] = None
        
        else:
            # Put index as key and intent name as value:
            intent_mapping[intent_class] = intent

    return augmented_utterances, np.array(utterance_classes), intent_mapping


def text_to_utterance(text):
    return {DATA: [{TEXT: text}]}
