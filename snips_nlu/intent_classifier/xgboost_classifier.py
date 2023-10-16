from __future__ import unicode_literals

import json
import logging
from builtins import range, str, zip
from pathlib import Path
import joblib
import os

from snips_nlu.common.log_utils import DifferedLoggingMessage, log_elapsed_time
from snips_nlu.common.utils import check_persisted_path, fitted_required, json_string
from snips_nlu.constants import LANGUAGE, RES_PROBA
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.exceptions import LoadingError, _EmptyDatasetUtterancesError
from snips_nlu.intent_classifier.featurizer import Featurizer
from snips_nlu.intent_classifier.intent_classifier import IntentClassifier
from snips_nlu.intent_classifier.log_reg_classifier_utils import build_training_data, text_to_utterance
from snips_nlu.pipeline.configs import XGBoostIntentClassifierConfig
from snips_nlu.result import intent_classification_result
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

logger = logging.getLogger(__name__)
DEBUG = False
TUNING = False


import re

class EntitiesRegexCompile():
    """Compile systems entities with regex rules"""

    def __init__(self):
        self.re_entities = re.compile(r'(@\w+(-\w+)*_\w+:\w+(-\w+)*)', re.I)
        #self.re_entities = re.compile(r'@[\w-]+_[\w]+:[\w-]+', re.I)
        self.re_questions = re.compile(r'[\[\]]', re.I)         # the two "\" are to escape and match "[]"
        self.ents_smartly = re.compile(r'\w+-\w+-\w+', re.I)
        self.ent_name = re.compile(r'@[a-zA-Z0-9-_]+_')
        self.find_entity_id = re.compile(r'_[a-zA-Z0-9]+:')
        self.smartly_sys_ent = re.compile(r'-\w+-\w+$', re.I)   # Regex to find Smartly's system entity format (e.g. SysEntName-fr-fr)
        self.smartly_custom_ent = re.compile(r'@[\w]+_[\w]+:[\w]+')  # Regex to match custom entities (excludes system entities)
        self.snips_format = re.compile(r'\[.+\]\(.+\)', re.I)  # general Snips entity format (custom AND system entities) => matches [entity](value) format
        self.null_excepts = re.compile(r'^(phone-number|email|url).+', re.I)
        self.arobase = re.compile(r'@')
        self.doublebare = re.compile(r'//')
        self.deuxpoints = re.compile(r':')
        self.point = re.compile(r'\.')
        self.email = re.compile(r'[\w.-]+@[\w]+.[\w]+', re.I)
        self.url = re.compile(r'(www\.[\w-]+\.[a-zA-Z]+)|([\w-]{2,}\.[a-zA-Z]{2,})|([a-zA-Z]+://[\w-]+.[a-zA-Z]+)', re.I)
        self.phone_number = re.compile(r'\b[0-9]{2,3}[-\s]{0,1}[0-9]{2,3}[-\s]{0,1}[0-9]{2,3}[-\s]{0,1}[0-9]{2,3}[-\s]{0,1}[0-9]{2,3}\b', re.I)


def smart_split(utterance):

    ''' Function to split utterance by white spaces AND by Snips entities
                => since a simple .split() ignores entities '''

    # Split by Snips entity format OR by white spaces (\s):
    s = re.compile(r'(\[.*?\]\(.*?\))|\s', re.I)
    space_ent_split = re.split(s, utterance)

    # Remove empty matches from list:
    space_ent_split = [element for element in space_ent_split if element not in [' ', '', None]]

    return space_ent_split

def load_smartly_emojis():
    """Load mapping Smartly emojis

    Returns:
        list: list of Snips / Smartly emojis
    """
    smartly_emojis = []
    with open('/common/data/web_emojis_data.json') as k:
        smartly_emojis = json.load(k)
    return smartly_emojis

def build_search_emojis(emojis_data):
    """build compile data for emojis

    Args:
        emojis_data (dict): emojis request data

    Returns:
        src_compile: regex compile fo entities
    """
    compile_string = "".join([k for k in emojis_data.keys()])
    compile_string = r"[{}]".format(compile_string)
    compile_emojis = re.compile(compile_string)
    return compile_emojis

emojis_data    = load_smartly_emojis()
compile_emojis = build_search_emojis(emojis_data)
global_regex = EntitiesRegexCompile()
re_ents      = global_regex.re_entities
re_quest     = global_regex.re_questions
re_entity_name = global_regex.ent_name
find_entity_id = global_regex.find_entity_id
ents_smtly   = global_regex.ents_smartly
re_custom_entity = global_regex.smartly_custom_ent
re_snips_format = global_regex.snips_format
re_email = global_regex.email
re_url = global_regex.url
re_phone_number = global_regex.phone_number

def preprocess_utterance(text):
    """Preprocess data for parsing

    Args:
        text (str): input data

    Returns:
        str: data format with fix values
    """
    text_remake = text
    if global_regex.arobase.search(text_remake): 
        text_remake = ' '.join([global_regex.arobase.sub(' arobase ', tok) if not re_snips_format.findall(tok) else tok for tok in smart_split(text_remake)])

    if global_regex.doublebare.search(text_remake): 
        text_remake = ' '.join([global_regex.doublebare.sub(' doublebare ', tok) if not re_snips_format.findall(tok) else tok for tok in smart_split(text_remake)])

    if global_regex.deuxpoints.search(text_remake): 
        text_remake = ' '.join([global_regex.deuxpoints.sub(' deuxpoints ', tok) if not re_snips_format.findall(tok) else tok for tok in smart_split(text_remake)])

    if not re.compile(r"\d+\.\d+").findall(text_remake):
        if global_regex.point.search(text_remake):
            text_remake = ' '.join([global_regex.point.sub(' point ', tok) if not re_snips_format.findall(tok) else tok  for tok in smart_split(text_remake)])


    if re.compile(r"\d+\.\d+").findall(text_remake):
        text_remake = text_remake.replace('.', ',')


    # Convert emojis to their names if they exist:
    # If a new-style emoji exists in the utterance:
    if compile_emojis.search(text_remake):

        # Find all emojis in utterance:
        emojis = compile_emojis.findall(text_remake)

        # Clean U+fe0f invisible character:
        emojis = [emoji for emoji in emojis if emoji != '️']

        # For each found emoji:
        for em in emojis:

            #print("emojis_data name: ", emojis_data[em]['name'])
            try:
                text_remake = re.sub(em, ' ' + emojis_data[em]['name'] + ' ' , text_remake)

                # Remove invisible variation character:
                text_remake = text_remake.replace('️', '')
            except:
                pass

        # Remove skin tone character:
        skin_tones = [' 🏻',' 🏼',' 🏽' ,' 🏾' ,' 🏿']

        for tone in skin_tones:
            text_remake = text_remake.replace(tone, '')

    # Remove double spaces if they exist:
    text_remake = text_remake.replace('  ', ' ')

    # Remove apostrophes (only if stop words are activated => to avoid multiplicity of definite articles for example):
    if False:
        text_remake = text_remake.replace("'", ' ').replace("’", ' ')

    # Remove diactritics (Zeus service preprocessing):
    text_remake = remove_diacritics(text_remake)

    return text_remake

# Function to remove diacritics:
def remove_diacritics(text):
    
    # Original text:
    text_remake = text

    # Remove diacritics:
    text_remake = re.sub(re.compile(r'Á|À|Â|Ä|A̧|Ą|Ⱥ|Ǎ|Ȧ|Ạ|Ā|Ã'), 'A', text_remake)
    text_remake = re.sub(re.compile(r'á|à|â|ä|a̧|ą|ⱥ|ǎ|ȧ|ạ|ā|ã'), 'a', text_remake)
    text_remake = re.sub(re.compile(r'É|È|Ê|Ë|Ȩ|Ę|Ɇ|Ě|Ė|Ẹ|Ē|Ẽ'), "E", text_remake)
    text_remake = re.sub(re.compile(r'é|è|ê|ë|ȩ|ę|ɇ|ě|ė|ẹ|ē|ẽ'), "e", text_remake)
    text_remake = re.sub(re.compile(r'Ć|C̀|Ĉ|C̈|Ç|C̨|Ȼ|Č|Ċ|C̣|C̄|C̃'), "C", text_remake)
    text_remake = re.sub(re.compile(r'ć|c̀|ĉ|c̈|ç|c̨|ȼ|č|ċ|c̣|c̄|c̃'), "c", text_remake)
    text_remake = re.sub(re.compile(r'Í|Ì|Î|Ï|I̧|Į|Ɨ|Ǐ|İ|Ị|Ī|Ĩ'), "I", text_remake)
    text_remake = re.sub(re.compile(r'í|ì|î|ï|i̧|į|ɨ|ǐ|i|ị|ī|ĩ'), "i", text_remake)
    text_remake = re.sub(re.compile(r'J́|J̀|Ĵ|J̈|J̧|J̨|Ɉ|J̌|J̇|J̣|J̄|J̃'), "J", text_remake)
    text_remake = re.sub(re.compile(r'j́|j̀|ĵ|j̈|j̧|j̨|ɉ|ǰ|j|j̣|j̄|j̃'), "j", text_remake)
    text_remake = re.sub(re.compile(r'Ĺ|L̀|L̂|L̈|Ļ|L̨|Ł|Ƚ|Ľ|L̇|Ḷ|L̄|L̃'), "L", text_remake)
    text_remake = re.sub(re.compile(r'ĺ|l̀|l̂|l̈|ļ|l̨|ł|ƚ|ľ|l̇|ḷ|l̄|l̃'), "l", text_remake)
    text_remake = re.sub(re.compile(r'Ń|Ǹ|N̂|N̈|Ņ|N̨|Ꞥ|Ň|Ṅ|Ṇ|N̄|Ñ'), "N", text_remake)
    text_remake = re.sub(re.compile(r'ń|ǹ|n̂|n̈|ņ|n̨|ꞥ|ň|ṅ|ṇ|n̄|ñ'), "n", text_remake)
    text_remake = re.sub(re.compile(r'Ó|Ò|Ô|Ö|O̧|Ǫ|Ø|Ɵ|Ǒ|Ȯ|Ọ|Ō|Õ'), "O", text_remake)
    text_remake = re.sub(re.compile(r'ó|ò|ô|ö|o̧|ǫ|ø|ɵ|ǒ|ȯ|ọ|ō|õ'), "o", text_remake)
    text_remake = re.sub(re.compile(r'Ś|S̀|Ŝ|S̈|Ş|S̨|Ꞩ|Š|Ṡ|Ṣ|S̄|S̃'), "S", text_remake)
    text_remake = re.sub(re.compile(r'ś|s̀|ŝ|s̈|ş|s̨|ꞩ|š|ṡ|ṣ|s̄|s̃'), "s", text_remake)
    text_remake = re.sub(re.compile(r'T́|T̀|T̂|T̈|Ţ|T̨|Ⱦ|Ŧ|Ť|Ṫ|Ṭ|T̄|T̃'), "T", text_remake)
    text_remake = re.sub(re.compile(r't́|t̀|t̂|ẗ|Ţ|ţ|t̨|ⱦ|ŧ|ť|ṫ|ṭ|t̄|t̃'), "t", text_remake)
    text_remake = re.sub(re.compile(r'Ú|Ù|Û|Ü|U̧|Ų|Ʉ|Ǔ|U̇|Ụ|Ū|Ũ'), "U", text_remake)
    text_remake = re.sub(re.compile(r'ú|ù|û|ü|u̧|ų|ʉ|ǔ|u̇|ụ|ū|ũ'), "u", text_remake)
    text_remake = re.sub(re.compile(r'Ý|Ỳ|Ŷ|Ÿ|Y̧|Y̨|Ɏ|Y̌|Ẏ|Ỵ|Ȳ|Ỹ'), "Y", text_remake)
    text_remake = re.sub(re.compile(r'ý|ỳ|ŷ|ÿ|y̧|y̨|ɏ|y̌|ẏ|ỵ|ȳ|ỹ'), "y", text_remake)
    text_remake = re.sub(re.compile(r'Ź|Z̀|Ẑ|Z̈|Z̧|Z̨|Ƶ|Ž|Ż|Ẓ|Z̄|Z̃'), "Z", text_remake)
    text_remake = re.sub(re.compile(r'ź|z̀|ẑ|z̈|z̧|z̨|ƶ|ž|ż|ẓ|z̄|z̃'), "z", text_remake)
    text_remake = re.sub(re.compile(r'Æ'), "AE", text_remake)
    text_remake = re.sub(re.compile(r'æ'), "ae", text_remake)
    text_remake = re.sub(re.compile(r'Œ'), "OE", text_remake)
    text_remake = re.sub(re.compile(r'œ'), "oe", text_remake)

    # Remove special characters and ponctuation: 
    text_remake = re.sub(re.compile(r'\?|!|¿|;|\{|\}|«|»|\"|\$|\<|\>|\*'), " ", text_remake)

    # Remove parenthesis and brackets ONLY if not Snips entity:
    text_remake = ' '.join([re.sub(re.compile(r'\[|\]|\(|\)'), " ", word) if not re_snips_format.findall(word)
                            else word
                            for word in smart_split(text_remake)])

    # Remove comma "," ONLY if it's not a float number:
    text_remake = ' '.join([re.sub(re.compile(r'(?<=[^0-9])(,)|(,)(?=[^0-9])'), " ", tok) if not re_snips_format.findall(tok) else tok for tok in smart_split(text_remake)])

    # Remove dot & comma only if it's not in a decimal number or in a Snips entity (else if float replace dot by comma):
    text_remake = ' '.join(
                            [word.replace('.', '').replace(',', '')
                            if not re.findall('[0-9]+[,|.][0-9]+', word) and not re_snips_format.findall(word)
                            else word if re_snips_format.findall(word)
                            else word.replace('.', ',')
                            for word in smart_split(text_remake)]
                            )

    # Remove hyphen IF not date OR negative number:
    date_neg_re = re.compile(r'\d{1,2}-\d{1,2}-\d{4}|\d{4}-\d{1,2}-\d{1,2}|-[0-9]+[,]+[0-9]+|-[0-9]+')
    text_remake = ' '.join([re.sub(re.compile(r'-'), " ", tok) if not re_snips_format.findall(tok) and not date_neg_re.findall(tok) else tok for tok in smart_split(text_remake)])

    # Remove multiple white spaces:
    text_remake = re.sub(re.compile(r'[ ]+'), " ", text_remake)

    # Remove trailing white spaces:
    text_remake = text_remake.strip()

    return text_remake


# We set tol to 1e-3 to silence the following warning with Python 2 (
# scikit-learn 0.20):
#
# FutureWarning: max_iter and tol parameters have been added in SGDClassifier
# in 0.19. If max_iter is set but tol is left unset, the default value for tol
# in 0.19 and 0.20 will be None (which is equivalent to -infinity, so it has no
# effect) but will change in 0.21 to 1e-3. Specify tol to silence this warning.


@IntentClassifier.register("xgboost_classifier")
class XGBoostIntentClassifier(IntentClassifier):
    """Intent classifier which uses a Random Forest Classifier underneath"""

    config_type = XGBoostIntentClassifierConfig

    def __init__(self, config=None, **shared):
        """The LogReg intent classifier can be configured by passing a
        :class:`.XGBoostIntentClassifier`"""
        super(XGBoostIntentClassifier, self).__init__(config, **shared)
        self.classifier = None
        self.intent_list = None
        self.featurizer = None

    @property
    def fitted(self):
        """Whether or not the intent classifier has already been fitted"""
        return self.intent_list is not None

    @log_elapsed_time(logger, logging.INFO,"XGBoostIntentClassifier in {elapsed_time}")
    def fit(self, dataset):
        """Fits the intent classifier with a valid Snips dataset

        Returns:
            :class:`XGBoostIntentClassifier`: The same instance, trained
        """
        from xgboost import XGBClassifier
        
        logger.info("Fitting XGBoostIntentClassifier...")
        dataset = validate_and_format_dataset(dataset) 
        self.load_resources_if_needed(dataset[LANGUAGE])
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)
        language = dataset[LANGUAGE]
        
        data_augmentation_config = self.config.data_augmentation_config


        '''df = pd.DataFrame(dataset['intents']).transpose()

        df['count'] = ''

        for i in range(len(df)):
            df['count'][df.index==df.index[i]] = len(df["utterances"][df.index==df.index[i]][df.index[i]])

        ones = df[df['count'] == 1]
        more_than_one = df[df['count'] > 1]

        from sklearn.model_selection import train_test_split
        more_train, more_test = train_test_split(more_than_one['utterances'].explode(),
                                                train_size=0.5,
                                                stratify = more_than_one['utterances'].explode().index,
                                                random_state=42)

        # Regroup the exploded list
        more_train = more_train.groupby(more_train.index).agg(list)

        # Add the stratified train to the Series with only one utterance per intent:
        final_train = pd.concat([more_train, ones['utterances']])

        # Convert to dataframe with series name as column name:
        final_train = pd.DataFrame(final_train)
        final_test = pd.DataFrame(more_test)
        final_test['utterances'] = final_test['utterances'].apply(lambda row: [row])

        # Convert to Snips compatible dict:
        dataset['intents'] = final_train.transpose().to_dict()'''

        # Build training data:
        utterances, classes, intent_list = build_training_data(dataset,
                                                               language,
                                                               data_augmentation_config,
                                                               self.resources,
                                                               self.random_state)
        # Store intent list:
        self.intent_list = intent_list

        '''dataset['intents'] = final_test.transpose().to_dict()        
        utterances_test, classes_test, intent_list_test = build_training_data(dataset,
                                                                                language,
                                                                                data_augmentation_config,
                                                                                self.resources,
                                                                                self.random_state)'''


        # If there is only one intent, we don't need to train a classifier (i.e. return instance):
        if len(self.intent_list) <= 1:
            return self

        # Instantiate the featurizer:
        self.featurizer = Featurizer(
                                    config=self.config.featurizer_config,
                                    builtin_entity_parser=self.builtin_entity_parser,
                                    custom_entity_parser=self.custom_entity_parser,
                                    resources=self.resources,
                                    random_state=self.random_state,
                                )
        
        # Set language:
        self.featurizer.language = language

        none_class = max(classes)

        # Fit the featurizer:
        try:
            x = self.featurizer.fit_transform(dataset, utterances, classes, none_class)
#           x_test = self.featurizer.fit_transform(dataset, utterances_test, classes_test, none_class)

            import pandas as pd
            df = pd.read_pickle("/snips_train/nfs_server/faycal_test_dataset.pickle").drop("expected_intent", axis=1) 

        except _EmptyDatasetUtterancesError:
            logger.warning("No (non-empty) utterances found in dataset")
            self.featurizer = None
            return self

        # Dimensionality reduction for debugging visualizations:
        if DEBUG:

            # Import PCA from SKlearn:
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE

            # Instantiate PCA:
            pca = PCA(n_components=2, random_state=self.random_state)

            # Fit transform PCA:
            self.pcad_x = pca.fit_transform(x.toarray())

            # Fit transform TSNE for 3 values of Perplexity: 5, N**1/2, 50
            self.tsned_x = {}
            
            # For each perplexity value in the following list:
            for perplexity in [5, len(classes)**(1/2), 50]:

                # Instantiate t-SNE with 2 dimensions:
                tsne = TSNE(n_components=2,
                            random_state=self.random_state,
                            perplexity = perplexity
                            )
                
                # Fit transform t-SNE and store the result in the tsned_x dict attribute:
                tsned_x = tsne.fit_transform(x.toarray())
                self.tsned_x[perplexity] = tsned_x

            # Persist the classes:
            self.classes = classes  

        # If hyperparameter tuning is disabled:
        if not TUNING:

            from sklearn.metrics import accuracy_score
            breakpoint()

            # Instantiate the classifier:
            class_weights = compute_class_weight("balanced", np.unique(classes), classes)
            weights_dict = dict(zip(np.unique(classes), class_weights))
            breakpoint()
            sample_weights = [weights_dict[class_idx] for class_idx in classes]
            #sample_weights = [class_weights[class_idx] for class_idx in y_train]

            #self.classifier = GradientBoostingClassifier(verbose=True, n_estimators=100)

            self.classifier = XGBClassifier(
                                        n_estimators = 100,
                                        objective = 'multi:softmax',
                                        n_jobs = os.cpu_count(),
                                        booster = 'gbtree',
                                        tree_method = 'hist',
                                        early_stopping_rounds = None,
                                        learning_rate = 0.01,
                                        random_state = self.random_state)

            # Fit the classifier normally:
            #self.classifier.fit(x_train, y_train, sample_weight=sample_weights)
            self.classifier.fit(x, classes,
                                eval_set=[(x, classes)],
                                verbose=True,
                                eval_metric= "mlogloss",
                                sample_weight= sample_weights)

            y_preds = self.classifier.predict(x_test)
            print(accuracy_score(y_preds, classes_test))
            breakpoint()

            
        # If tuning is enabled:
        else:

            import optuna
            from sklearn.model_selection import StratifiedKFold
            from xgboost import cv, DMatrix
            import optuna.integration.xgboost as xgb_integration
            from sklearn.metrics import f1_score

            def objective(trial):

                # Define hyperparameter search space
                params = {
                    'objective':'multi:softprob',
                    'booster':'gbtree',
                    'tree_method' : 'hist',
                    'n_jobs': 1,
                    'num_class' : len(classes),
                    'random_state': 42,
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.1, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
                    'gamma': trial.suggest_float('gamma', 0.01, 10.0, log=True)
                }

                # Custom weighted F1 score metric for multiclass classification
                def weighted_f1_score(preds, dtrain):
                    y_true = dtrain.get_label()
                    y_pred = np.argmax(preds, axis=1)
                    f1 = f1_score(y_true, y_pred, average = 'weighted')
                    return 'weighted-f1-score', f1

                # Stratified Cross-Validation using xgb.cv() with XGBoostPruningCallback
                pruning_callback = xgb_integration.XGBoostPruningCallback(trial, 'test-weighted-f1-score')

                # Create stratified folds:
                strat_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                # Run cross validation:
                cv_results = cv(params,
                                dtrain = DMatrix(x, label=classes),
                                num_boost_round = 100,
                                folds = strat_folds,
                                early_stopping_rounds = 10,
                                maximize = True,
                                custom_metric = weighted_f1_score,
                                callbacks = [pruning_callback],
                                seed = 42,
                                verbose_eval = False)


                # Use the mean value across all folds as the objective:
                return np.mean(cv_results['test-weighted-f1-score-mean'])

            # Define Optuna study
            study = optuna.create_study(direction = 'maximize',
                                        pruner = optuna.pruners.HyperbandPruner())

            # Start optimization process:
            study.optimize(objective,
                           n_trials = 100,
                           n_jobs = max(os.cpu_count()-2, 2),
                           show_progress_bar = True,
                           gc_after_trial = True)

            # Get the best parameters
            best_params = study.best_params

            # Print the best hyperparameters and score
            best_trial = study.best_trial
            print('Best trial:')
            print('Value: {}'.format(best_trial.value))
            print('Params: ')
            for key, value in best_trial.params.items():
                print('    {}: {}'.format(key, value))

            # Retrain XGBClassifier on the whole dataset with the best parameters:
            self.classifier = XGBClassifier(**best_params,
                                    objective ='multi:softmax',
                                    n_jobs = os.cpu_count(),
                                    booster='gbtree',
                                    tree_method = 'hist',
                                    random_state = self.random_state)
            
            # Fit final model:
            self.classifier.fit(x, classes)

        logger.debug("%s", DifferedLoggingMessage(self.log_best_features))
        return self


    # Get best intent => uses get_intents to get top best intents and then takes the first one:
    @fitted_required
    def get_intent(self, text, intents_filter=None): 
        """Performs intent classification on the provided *text*

        Args:
            text (str): Input
            intents_filter (str or list of str): When defined, it will find
                the most likely intent among the list, otherwise it will use
                the whole list of intents defined in the dataset

        Returns:
            dict or None: The most likely intent along with its probability or
            *None* if no intent was found

        Raises:
            :class:`snips_nlu.exceptions.NotTrained`: When the intent
                classifier is not fitted

        """
        return self._get_intents(text, intents_filter)[0]

    @fitted_required
    def get_intents(self, text):
        """Performs intent classification on the provided *text* and returns
        the list of intents ordered by decreasing probability

        The length of the returned list is exactly the number of intents in the
        dataset + 1 for the None intent

        Raises:
            :class:`snips_nlu.exceptions.NotTrained`: when the intent
                classifier is not fitted
        """
        return self._get_intents(text, intents_filter=None)


    def _get_intents(self, text, intents_filter):
        """Performs intent classification on the provided *text* and returns
        the list of intents ordered by decreasing probability"""

        if isinstance(intents_filter, str):
            intents_filter = {intents_filter}

        elif isinstance(intents_filter, list):
            intents_filter = set(intents_filter)

        if not text or not self.intent_list or not self.featurizer:
            """ The function intent_classification_result() simply formats whatever
            intent name and probability it receives into a dict.

            Example:

            intent_classification_result("GetWeather", 0.93)
            Returns : {'intentName': 'GetWeather', 'probability': 0.93}
            """

            # If no text or no intent list or no featurizer, return None with 100% probability:
            results = [intent_classification_result(None, 1.0)]

            # Append the rest of the intents with 0% probability:
            results += [intent_classification_result(i, 0.0) for i in self.intent_list if i is not None]

            return results

        # If only one intent, return it with 100% probability:
        if len(self.intent_list) == 1:
            return [intent_classification_result(self.intent_list[0], 1.0)]

        # pylint: disable=C0103
        # Transform the text into a vector of features:
        X = self.featurizer.transform([text_to_utterance(text)])

        # pylint: enable=C0103
        proba_vec = self.classifier.predict_proba(X).astype("float64")

        logger.debug("%s", DifferedLoggingMessage(self.log_activation_weights, text, X))

        results = [
                    intent_classification_result(i, proba)
                    for i, proba in zip(self.intent_list, proba_vec[0])
                    if intents_filter is None or i is None or i in intents_filter
               ]

        return sorted(results, key=lambda res: -res[RES_PROBA])
    

    @check_persisted_path
    def persist(self, path):
        """Persists the object at the given path"""

        # Make new directory with unique path:
        path.mkdir()
        
        # Persist featurizer config:
        if self.featurizer is not None:
            featurizer = "featurizer"
            featurizer_path = path / featurizer
            self.featurizer.persist(featurizer_path)


        # Classifier's config dict:
        self_as_dict = {
            "config": self.config.to_dict(),
            "intent_list": self.intent_list,
            "featurizer": featurizer
        }

        # Convert config to JSON:
        classifier_json = json_string(self_as_dict)

        # Persist the JSON:
        with (path / "intent_classifier.json").open(mode="w", encoding="utf8") as f:
            f.write(classifier_json)

        # Add metadata:
        self.persist_metadata(path)

        # Persist the classifier:
        joblib.dump(self.classifier, str(path / "xgboost_model.joblib"), compress=3)

        # Persist TF-IDF vectorizer:
        #joblib.dump(self.featurizer.vectorizer._sklearn_tfidf_vectorizer, str(path / "tfidf.joblib"), compress=3)

        # t-SNE and PCA embeddings if debug mode is on:
        if DEBUG:

            # Use matplotlib to plot the t-SNE for each perplexity:
            for perplexity in self.tsned_x.keys():
                    
                    import matplotlib.pyplot as plt

                    # Instantiate a figure:
                    fig = plt.figure(figsize=(8, 8))
        
                    # Plot the t-SNE:
                    plt.scatter(self.tsned_x[perplexity][:, 0],
                                self.tsned_x[perplexity][:, 1],
                                c=self.classes,
                                cmap=plt.cm.get_cmap("jet", len(self.intent_list)), s=10, alpha=0.7)
        
                    # Add a colorbar:
                    cbar = plt.colorbar(ticks=range(len(self.intent_list)))
                    cbar.ax.set_yticklabels(self.intent_list)
        
                    # Add a title:
                    plt.title("t-SNE for perplexity = %s" % perplexity)

                    # Save the figure:
                    plt.savefig(str(path / ("tsne_perplexity_%s.png" % perplexity)))

            # Plot PCA:
            fig = plt.figure(figsize=(8, 8))
            plt.scatter(self.pcad_x[:, 0],
                        self.pcad_x[:, 1],
                        c=self.classes,
                        cmap=plt.cm.get_cmap("jet", len(self.intent_list)), s=10, alpha=0.7)
            
            # Add a colorbar:
            cbar = plt.colorbar(ticks=range(len(self.intent_list)))
            cbar.ax.set_yticklabels(self.intent_list)

            # Add a title:
            plt.title("PCA")

            # Save the figure:
            plt.savefig(str(path / "pca.png"))

            # Persist the tsned_x dict:
            joblib.dump(self.tsned_x, str(path / "tsned_x.joblib"), compress=3)

            # Persist the classes:
            joblib.dump(self.classes, str(path / "classes.joblib"), compress=3)


    @classmethod
    def from_path(cls, path, **shared):
        """Loads a :class:`RandForIntentClassifier` instance from a path

        The data at the given path must have been generated using
        :func:`~RandForIntentClassifier.persist`
        """

        path = Path(path)
        model_path = path / "intent_classifier.json"
        if not model_path.exists():
            raise LoadingError("Missing intent classifier model file: %s" % model_path.name)

        with model_path.open(encoding="utf8") as f:
            model_dict = json.load(f)

        # Create the classifier
        config = XGBoostIntentClassifierConfig.from_dict(model_dict["config"])
        intent_classifier = cls(config=config, **shared)
        intent_classifier.intent_list = model_dict['intent_list']

        # Create the underlying SGD classifier 
        intent_classifier.classifier = joblib.load(str(path / "xgboost_model.joblib"))

        # Add the featurizer
        featurizer = model_dict['featurizer']
        if featurizer is not None:
            featurizer_path = path / featurizer
            intent_classifier.featurizer = Featurizer.from_path(featurizer_path, **shared)

        return intent_classifier

    def log_best_features(self, top_n=50):
        import numpy as np

        if not hasattr(self.featurizer, "feature_index_to_feature_name"):
            return None

        log = "Top {} features weights by intent:".format(top_n)
        index_to_feature = self.featurizer.feature_index_to_feature_name
        for intent_ix in range(self.classifier.coef_.shape[0]):
            intent_name = self.intent_list[intent_ix]
            log += "\n\n\nFor intent {}\n".format(intent_name)
            top_features_idx = np.argsort(
                np.absolute(self.classifier.coef_[intent_ix]))[::-1][:top_n]
            for feature_ix in top_features_idx:
                feature_name = index_to_feature[feature_ix]
                feature_weight = self.classifier.coef_[intent_ix, feature_ix]
                log += "\n{} -> {}".format(feature_name, feature_weight)
        return log

    def log_activation_weights(self, text, x, top_n=50):
        import numpy as np

        if not hasattr(self.featurizer, "feature_index_to_feature_name"):
            return None

        log = "\n\nTop {} feature activations for: \"{}\":\n".format(
            top_n, text)
        activations = np.multiply(
            self.classifier.coef_, np.asarray(x.todense()))
        abs_activation = np.absolute(activations).flatten().squeeze()

        if top_n > activations.size:
            top_n = activations.size

        top_n_activations_ix = np.argpartition(abs_activation, -top_n,
                                               axis=None)[-top_n:]
        top_n_activations_ix = np.unravel_index(
            top_n_activations_ix, activations.shape)

        index_to_feature = self.featurizer.feature_index_to_feature_name
        features_intent_and_activation = [
            (self.intent_list[i], index_to_feature[f], activations[i, f])
            for i, f in zip(*top_n_activations_ix)]

        features_intent_and_activation = sorted(
            features_intent_and_activation, key=lambda x: abs(x[2]),
            reverse=True)

        for intent, feature, activation in features_intent_and_activation:
            log += "\n\n\"{}\" -> ({}, {:.2f})".format(
                intent, feature, float(activation))
        log += "\n\n"
        return log
