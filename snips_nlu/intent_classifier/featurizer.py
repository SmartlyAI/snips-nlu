from __future__ import division, unicode_literals

import json
from builtins import str, zip
from copy import deepcopy
from pathlib import Path
from scipy import sparse as sp
import joblib

from future.utils import iteritems

from snips_nlu.common.utils import (
    json_string, fitted_required, replace_entities_with_placeholders,
    check_persisted_path)
from snips_nlu.constants import (
    DATA, ENTITY, ENTITY_KIND, LANGUAGE, NGRAM, TEXT, ENTITIES)
from snips_nlu.dataset import get_text_from_chunks, validate_and_format_dataset
from snips_nlu.entity_parser.builtin_entity_parser import (
    is_builtin_entity)
from snips_nlu.exceptions import (_EmptyDatasetUtterancesError, LoadingError)
from snips_nlu.languages import get_default_sep
from snips_nlu.pipeline.configs import FeaturizerConfig
from snips_nlu.pipeline.configs.intent_classifier import (
    CooccurrenceVectorizerConfig, VectorizerConfig)
from snips_nlu.pipeline.processing_unit import ProcessingUnit
from snips_nlu.preprocessing import stem, tokenize_light
from snips_nlu.resources import get_stop_words, get_word_cluster
from snips_nlu.slot_filler.features_utils import get_all_ngrams


@ProcessingUnit.register("featurizer")
class Featurizer(ProcessingUnit):
    """Feature extractor for text classification relying on ngrams tfidf and
    optionally word cooccurrences features"""

    config_type = FeaturizerConfig

    def __init__(self, config=None, **shared):
        super(Featurizer, self).__init__(config, **shared)
        self.language = None
        self.vectorizer = None
        self.cooccurrence_vectorizer = None

    @property
    def fitted(self):
        if not self.vectorizer or not self.vectorizer.vocabulary:
            return False
        return True

    @property
    def feature_index_to_feature_name(self):
        """Maps the feature index of the feature matrix to printable features
        names. Mainly useful for debug.

        Returns:
            dict: a dict mapping feature indices to printable features names
        """
        if not self.fitted:
            return dict()

        index = {
            i: "ngram:%s" % ng
            for ng, i in iteritems(self.vectorizer.vocabulary)
        }
        num_ng = len(index)
        if self.cooccurrence_vectorizer is not None:
            for word_pair, j in iteritems(
                    self.cooccurrence_vectorizer.word_pairs):
                index[j + num_ng] = "pair:%s+%s" % (word_pair[0], word_pair[1])
        return index


    # Method that calls the vectorizers' fit_transform() methods:
    def fit_transform(self, dataset, utterances, classes, none_class):
        import scipy.sparse as sp

        dataset = validate_and_format_dataset(dataset)
        self.language = dataset[LANGUAGE]

        # Generator yielding utterances as Strings:
        utterances_texts = (get_text_from_chunks(u[DATA]) for u in utterances)

        # If there are no utterances:
        if not any(tokenize_light(q, self.language) for q in utterances_texts):
            raise _EmptyDatasetUtterancesError("Tokenized utterances are empty")

        # Fit TF-IDF vectorizer AND does khi-2 feature selection => returns sparse matrix of columns: significant words, rows: all utterances
        x_vectorized = self._fit_transform_vectorizer(utterances, classes, dataset, vectorizer_name = self.config.vectorizer_config.unit_name)

        # X: explanatory variables/input:
        x = x_vectorized

        # Only fit the Cooccurrence Vectorizer if "added_cooccurrence_feature_ratio" exists in the config:
        if self.config.added_cooccurrence_feature_ratio:

            # Fit cooccurrence vector:
            self._fit_cooccurrence_vectorizer(utterances, classes, none_class, dataset)
            x_cooccurrence = self.cooccurrence_vectorizer.transform(utterances)

            # Append cooccurrence columns to the prior TF-IDF columns:
            x = sp.hstack((x_vectorized, x_cooccurrence))

        return x
    

      # Same as fit_transform() but returns the fitted vectorizer instead of transformed data:
    def fit(self, dataset, utterances, classes, none_class):
        self.fit_transform(dataset, utterances, classes, none_class)
        return self
    

    # Featurizer's transform() calls the vectorizer's transform() method:
    def transform(self, utterances):
        import scipy.sparse as sp

        x = self.vectorizer.transform(utterances)
        if self.cooccurrence_vectorizer:
            x_cooccurrence = self.cooccurrence_vectorizer.transform(utterances)
            x = sp.hstack((x, x_cooccurrence))
        return x

    
    def _fit_transform_vectorizer(self, x, y, dataset, vectorizer_name):

        # Get required vectorizer through Factory:
        self.vectorizer = VectorizerFactory(vectorizer_name, self.config)

        # We can't return x_tfidf[:best_tfidf_features] because of the
        # normalization in the transform of the tfidf_vectorizer
        # this would lead to inconsistent result between: fit_transform(x, y)
        # and fit(x, y).transform(x)

        #! In reality, the sparse result returned is given independantly by the following ".transform(x)"
        #! the ".fit_transform(x,dataset)" above is only used to select the best n-grams

        # TODO replace transform() with fit_transform()
        return self.vectorizer.fit_transform(x, y, dataset)
    

    def _fit_cooccurrence_vectorizer(self, x, classes, none_class, dataset):
        import numpy as np
        from sklearn.feature_selection import chi2

        non_null_x = (d for d, c in zip(x, classes) if c != none_class)

        self.cooccurrence_vectorizer = CooccurrenceVectorizer(
            config=self.config.cooccurrence_vectorizer_config,
            builtin_entity_parser=self.builtin_entity_parser,
            custom_entity_parser=self.custom_entity_parser,
            resources=self.resources,
            random_state=self.random_state,
        )

        # Compute Word Pairs and fit cooccurrence vectorizer based on word-pair occurrences:
        x_cooccurrence = self.cooccurrence_vectorizer.fit(non_null_x, dataset).transform(x)

        # If there are no word pairs just return the featurizer (nothing to fit):
        if not self.cooccurrence_vectorizer.word_pairs:
            return self

        # Feature selection for coccurrences:
        _, pval = chi2(x_cooccurrence, classes)
        
        try:
            voc = len(self.vectorizer.tfidf_vectorizer.idf_diag)

        except:
            voc = round(len(self.cooccurrence_vectorizer.word_pairs)/2)

        top_k = int(self.config.added_cooccurrence_feature_ratio * voc)

        # No selection if k is greater or equal than the number of word pairs (keep everything)
        if top_k >= len(self.cooccurrence_vectorizer.word_pairs):
            return self

        top_k_cooccurrence_ix = np.argpartition(pval, top_k - 1, axis=None)[:top_k]

        top_k_cooccurrence_ix = set(top_k_cooccurrence_ix)
        top_word_pairs = [pair for pair, i in iteritems(self.cooccurrence_vectorizer.word_pairs) if i in top_k_cooccurrence_ix]
        
        # Only keep most significant word pairs (i.e. cooccurrences):
        self.cooccurrence_vectorizer.limit_word_pairs(top_word_pairs)
        return self

    @check_persisted_path
    def persist(self, path):
        path.mkdir()

        # Persist the vectorizers
        tfidf_vectorizer = None
        if self.vectorizer:
            tfidf_vectorizer = self.vectorizer.unit_name
            tfidf_vectorizer_path = path / tfidf_vectorizer
            self.vectorizer.persist(tfidf_vectorizer_path)

        cooccurrence_vectorizer = None
        if self.cooccurrence_vectorizer:
            cooccurrence_vectorizer = self.cooccurrence_vectorizer.unit_name
            cooccurrence_vectorizer_path = path / cooccurrence_vectorizer
            self.cooccurrence_vectorizer.persist(cooccurrence_vectorizer_path)

        # Persist main object
        self_as_dict = {
            "language_code": self.language,
            "tfidf_vectorizer": tfidf_vectorizer,
            "cooccurrence_vectorizer": cooccurrence_vectorizer,
            "config": self.config.to_dict()
        }

        featurizer_path = path / "featurizer.json"
        with featurizer_path.open("w", encoding="utf-8") as f:
            f.write(json_string(self_as_dict))

        # Persist metadata
        self.persist_metadata(path)


    # Load featurizer from a path for predictions:
    @classmethod
    def from_path(cls, path, **shared):

        path = Path(path)

        model_path = path / "featurizer.json"

        if not model_path.exists():
            raise LoadingError("Missing featurizer model file: %s" % model_path.name)
        
        with model_path.open("r", encoding="utf-8") as f:
            featurizer_dict = json.load(f)

        featurizer_config = featurizer_dict["config"]
        featurizer = cls(featurizer_config, **shared)

        featurizer.language = featurizer_dict["language_code"]

        #! TF-IDF, FastText, etc.
        try:
            vectorizer_name = featurizer_dict['config']['vectorizer_config']['unit_name']

        except:
            vectorizer_name = featurizer_dict['config']['tfidf_vectorizer_config']['unit_name']

        # Load vectorizer:
        featurizer.vectorizer = cls.load_vectorizer(vectorizer_name, path, featurizer_dict, shared)
        
        # Load cooccurrence vectorizer:
        cooccurrence_vectorizer = featurizer_dict["cooccurrence_vectorizer"]

        if cooccurrence_vectorizer:
            vectorizer_path = path / featurizer_dict["cooccurrence_vectorizer"]
            cooccurrence_vectorizer = CooccurrenceVectorizer.from_path(vectorizer_path, **shared)

        featurizer.cooccurrence_vectorizer = cooccurrence_vectorizer

        return featurizer
    
    @classmethod
    def load_vectorizer(cls, vectorizer_name, path, featurizer_dict, shared):

        # TF-IDF:
        if vectorizer_name == "tfidf_vectorizer":
            vectorizer_path = path / featurizer_dict["tfidf_vectorizer"]
            tfidf_vectorizer = TfidfVectorizer.from_path(vectorizer_path, **shared)
            return tfidf_vectorizer
        
        # FastText:
        elif vectorizer_name == "fasttext_vectorizer":
            return FastTextVectorizer()
        
        # SBERT:
        elif vectorizer_name == "sbert_vectorizer":
            return SBERTVectorizer()        
        
        # Raise error if unknown vectorizer:
        else:
            raise ValueError("Unknown vectorizer: %s" % vectorizer_name)

    

@ProcessingUnit.register("tfidf_vectorizer")
class TfidfVectorizer(ProcessingUnit):
    """Wrapper of the scikit-learn TfidfVectorizer"""

    config_type = VectorizerConfig

    def __init__(self, config=None, **shared):
        super(TfidfVectorizer, self).__init__(config, **shared)
        self._sklearn_tfidf_vectorizer = None
        self._language = None
        self.builtin_entity_scope = None

    # Fit the "IDF" part of the vectorizer:
    def fit(self, x, dataset):
        """Fits the idf of the vectorizer on the given utterances after
        enriching them with builtin entities matches, custom entities matches
        and the potential word clusters matches

        Args:
            x (list of dict): list of utterances
            dataset (dict): dataset from which x was extracted (needed to
                extract the language and the builtin entity scope)

        Returns:
            :class:`.TfidfVectorizer`: The fitted vectorizer
        """
        self.load_resources_if_needed(dataset[LANGUAGE])
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)

        self._language = dataset[LANGUAGE]

        # Initialize Sklearn vectorizer and define the tokenizer within it:
        self._init_sklearn_vectorizer(self._language)
        self.builtin_entity_scope = set(
            e for e in dataset[ENTITIES] if is_builtin_entity(e))
        preprocessed_data = self._preprocess(x)
        utterances = [
            self._enrich_utterance(u, builtin_ents, custom_ents, w_clusters)
            for u, builtin_ents, custom_ents, w_clusters
            in zip(*preprocessed_data)
        ]
        return self._sklearn_tfidf_vectorizer.fit(utterances)

    #! Fit the IDF part and add entities:
    # TODO: rename this to transform and remove "dataset" argument
    def transform(self, x):
        """Fits the idf of the vectorizer on the given utterances after
        enriching them with builtin entities matches, custom entities matches
        and the potential word clusters matches.
        Returns the featurized utterances.

        Args:
            x (list of dict): list of utterances
            dataset (dict): dataset from which x was extracted (needed to
                extract the language and the builtin entity scope)

        Returns:
            :class:`.scipy.sparse.csr_matrix`: A sparse matrix X of shape
            (len(x), len(self.vocabulary)) where X[i, j] contains tfdif of
            the ngram of index j of the vocabulary in the utterance i
        """

        # Formatted data: tuple of => utterances, system entities, custom entities and word clusters
        preprocessed_data = self._preprocess(x)

        # List of utterances => with some entities-related processing:
        utterances = [
            self._enrich_utterance(u, builtin_ents, custom_ents, w_clusters)
            for u, builtin_ents, custom_ents, w_clusters
            in zip(*preprocessed_data)
        ]

        #! Calls the fit_transform() method of the Sklearn vectorizer:
        # Return TF-IDF sparse CSR matrix => rows are documents (i.e. utterances) and columns are *unique* words (i.e. vocabulary):
        # TODO: replace fit_transform with transform()
        return self._sklearn_tfidf_vectorizer.transform(utterances)

    @property
    def fitted(self):
        return self._sklearn_tfidf_vectorizer is not None and hasattr(self._sklearn_tfidf_vectorizer, "vocabulary_")

    #@fitted_required
    # TODO: rename this to "fit_transform" but keep same arguments
    def fit_transform(self, x, y, dataset):
        """Featurizes the given utterances after enriching them with builtin
        entities matches, custom entities matches and the potential word
        clusters matches

        Args:
            x (list of dict): list of utterances

        Returns:
            :class:`.scipy.sparse.csr_matrix`: A sparse matrix X of shape
            (len(x), len(self.vocabulary)) where X[i, j] contains tfdif of
            the ngram of index j of the vocabulary in the utterance i

        Raises:
            NotTrained: when the vectorizer is not fitted:
        """

        from sklearn.feature_selection import chi2

        # Instantiate TF-IDF:
        self.tfidf_vectorizer = TfidfVectorizer(config=self.config,
                                                builtin_entity_parser=self.builtin_entity_parser,
                                                custom_entity_parser=self.custom_entity_parser,
                                                resources=self.resources,
                                                random_state=self.random_state,
                                                )
        
        # Fit slot parsers and get resources if needed:
        self.load_resources_if_needed(dataset[LANGUAGE])
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)

        # Get language:
        self._language = dataset[LANGUAGE]

        # Initialize Sklearn TF-IDF and define the tokenizer within it:
        self._init_sklearn_vectorizer(self._language)

        # Set builtin entity scope (used in "_preprocess()" method):
        self.builtin_entity_scope = set(e for e in dataset[ENTITIES] if is_builtin_entity(e))
        
        #! Transform the utterances into a TF-IDF matrix:
        utterances = [self._enrich_utterance(u, builtin_ents, custom_ents, w_clusters)
                      for u, builtin_ents, custom_ents, w_clusters
                      in zip(*self._preprocess(x))]
        
        # Fit the IDF:
        x_tfidf = self._sklearn_tfidf_vectorizer.fit_transform(utterances)

        # If list of unique words is empty:
        if not self.vocabulary:
            raise _EmptyDatasetUtterancesError("Dataset is empty or with empty utterances")

        # Khi-2 feature selection:
        _, tfidf_pval = chi2(x_tfidf, y)

        # Set of features that fall below p-value:
        best_tfidf_features = set(i for i, v in enumerate(tfidf_pval) if v < self.config.pvalue_threshold)

        # If no features are significant:
        if not best_tfidf_features:
            best_tfidf_features = set(idx for idx, val in enumerate(tfidf_pval) if val == tfidf_pval.min())

        # Get the ngrams corresponding to the best (i.e. significant) features:
        best_ngrams = [ng for ng, i in iteritems(self.vocabulary) if i in best_tfidf_features]

        # Only keep the significant columns (i.e. features/words):
        #! result is set in the class attribute "self._tfidf_vectorizer.vocabulary_"
        self.limit_vocabulary(best_ngrams)

        #! Use Sklearn's TF-IDF vectorizer method:
        return self._sklearn_tfidf_vectorizer.transform(utterances)


    def _preprocess(self, utterances):
        normalized_utterances = deepcopy(utterances)

        try: 
            use_stemming = self.config.vectorizer_config.use_stemming
            word_clusters_names = self.config.vectorizer_config.word_clusters_name


        except: 
            use_stemming = self.config.use_stemming
            word_clusters_names = self.config.word_clusters_name

        for u in normalized_utterances:
            nb_chunks = len(u[DATA])
            for i, chunk in enumerate(u[DATA]):
                chunk[TEXT] = _normalize_stem(chunk[TEXT], self.language, self.resources, use_stemming)
                if i < nb_chunks - 1:
                    chunk[TEXT] += " "

        # Extract builtin entities on unormalized utterances
        builtin_ents = [
            self.builtin_entity_parser.parse(
                get_text_from_chunks(u[DATA]),
                self.builtin_entity_scope, use_cache=True)
            for u in utterances
        ]
        # Extract builtin entities on normalized utterances
        custom_ents = [
            self.custom_entity_parser.parse(
                get_text_from_chunks(u[DATA]), use_cache=True)
            for u in normalized_utterances
        ]
        if word_clusters_names:
            # Extract world clusters on unormalized utterances
            original_utterances_text = [get_text_from_chunks(u[DATA])
                                        for u in utterances]
            w_clusters = [
                _get_word_cluster_features(
                    tokenize_light(u.lower(), self.language),
                    word_clusters_names,
                    self.resources)
                for u in original_utterances_text
            ]
        else:
            w_clusters = [None for _ in normalized_utterances]

        return normalized_utterances, builtin_ents, custom_ents, w_clusters


    # Enriches the utterance with builtin entities, custom entities and word clusters:
    def _enrich_utterance(self, utterance, builtin_entities, custom_entities,word_clusters):
        
        custom_entities_features = [_entity_name_to_feature(e[ENTITY_KIND], self.language) for e in custom_entities]

        builtin_entities_features = [_builtin_entity_to_feature(ent[ENTITY_KIND], self.language) for ent in builtin_entities]

        # We remove values of builtin slots from the utterance to avoid
        # learning specific samples such as '42' or 'tomorrow'
        filtered_tokens = [
            chunk[TEXT] for chunk in utterance[DATA]
            if ENTITY not in chunk or not is_builtin_entity(chunk[ENTITY])
        ]

        features = get_default_sep(self.language).join(filtered_tokens)

        if builtin_entities_features:
            features += " " + " ".join(sorted(builtin_entities_features))
        if custom_entities_features:
            features += " " + " ".join(sorted(custom_entities_features))
        if word_clusters:
            features += " " + " ".join(sorted(word_clusters))

        return features

    @property
    def language(self):
        # Create this getter to prevent the language from being set elsewhere
        # than in the fit
        return self._language

    @property
    def vocabulary(self):
        if self._sklearn_tfidf_vectorizer and hasattr(
                self._sklearn_tfidf_vectorizer, "vocabulary_"):
            return self._sklearn_tfidf_vectorizer.vocabulary_
        return None

    #@fitted_required
    def limit_vocabulary(self, ngrams):
        """Restrict the vectorizer vocabulary to the given ngrams

        Args:
            ngrams (iterable of str or tuples of str): ngrams to keep

        Returns:
            :class:`.TfidfVectorizer`: The vectorizer with limited vocabulary
        """
        import scipy.sparse as sp

        ngrams = set(ngrams)
        vocab = self.vocabulary
        existing_ngrams = set(vocab)
        extra_values = ngrams - existing_ngrams

        if extra_values:
            raise ValueError("Invalid ngrams %s, expected values in word_pairs" % sorted(extra_values))

        new_ngrams, new_index = zip(*sorted((ng, vocab[ng]) for ng in ngrams))

        self._sklearn_tfidf_vectorizer.vocabulary_ = {
            ng: new_i for new_i, ng in enumerate(new_ngrams)
        }
        # pylint: disable=protected-access
        # The new_idf_data is valid because the previous _idf_diag was indexed
        # with sorted ngrams and new_index is also indexed with sorted ngrams

        new_idf_data = self._sklearn_tfidf_vectorizer._tfidf._idf_diag.data[list(new_index)]

        self._sklearn_tfidf_vectorizer._tfidf._idf_diag = sp.spdiags(new_idf_data, diags=0, m=len(new_index), n=len(new_index), format="csr")
        # pylint: enable=protected-access

        #! This return is redundant => the returned value is
        #! not used because the result is set as an attribute
        return self

    @property
    def idf_diag(self):
        if self._sklearn_tfidf_vectorizer and hasattr(self._sklearn_tfidf_vectorizer, "vocabulary_"):
            return self._sklearn_tfidf_vectorizer.idf_
        return None

    def _init_sklearn_vectorizer(self, language):
        from sklearn.feature_extraction.text import (TfidfVectorizer as SklearnTfidfVectorizer)

        self._sklearn_tfidf_vectorizer = SklearnTfidfVectorizer(tokenizer=lambda x: tokenize_light(x, language))
        return self

    @check_persisted_path
    def persist(self, path):
        path.mkdir()

        vectorizer_ = None
        if self._sklearn_tfidf_vectorizer is not None:
            vocab = {k: int(v) for k, v in iteritems(self.vocabulary)}
            idf_diag = self.idf_diag.tolist()
            vectorizer_ = {
                "vocab": vocab,
                "idf_diag": idf_diag
            }

        builtin_entity_scope = None
        if self.builtin_entity_scope is not None:
            builtin_entity_scope = list(self.builtin_entity_scope)

        self_as_dict = {
            "vectorizer": vectorizer_,
            "language_code": self.language,
            "builtin_entity_scope": builtin_entity_scope,
            "config": self.config.to_dict(),
        }

        vectorizer_path = path / "vectorizer.json"
        with vectorizer_path.open("w", encoding="utf-8") as f:
            f.write(json_string(self_as_dict))
        self.persist_metadata(path)

    @classmethod
    # pylint: disable=W0212
    def from_path(cls, path, **shared):
        import numpy as np
        import scipy.sparse as sp
        from sklearn.feature_extraction.text import (
            TfidfTransformer, TfidfVectorizer as SklearnTfidfVectorizer)

        path = Path(path)

        model_path = path / "vectorizer.json"
        if not model_path.exists():
            raise LoadingError("Missing vectorizer model file: %s"
                               % model_path.name)
        with model_path.open("r", encoding="utf-8") as f:
            vectorizer_dict = json.load(f)

        vectorizer = cls(vectorizer_dict["config"], **shared)
        vectorizer._language = vectorizer_dict["language_code"]

        builtin_entity_scope = vectorizer_dict["builtin_entity_scope"]
        if builtin_entity_scope is not None:
            builtin_entity_scope = set(builtin_entity_scope)
        vectorizer.builtin_entity_scope = builtin_entity_scope

        vectorizer_ = vectorizer_dict["vectorizer"]
        if vectorizer_:
            vocab = vectorizer_["vocab"]
            idf_diag_data = vectorizer_["idf_diag"]
            idf_diag_data = np.array(idf_diag_data)

            idf_diag_shape = (len(idf_diag_data), len(idf_diag_data))
            row = list(range(idf_diag_shape[0]))
            col = list(range(idf_diag_shape[0]))
            idf_diag = sp.csr_matrix(
                (idf_diag_data, (row, col)), shape=idf_diag_shape)

            tfidf_transformer = TfidfTransformer()
            tfidf_transformer._idf_diag = idf_diag

            vectorizer_ = SklearnTfidfVectorizer(
                tokenizer=lambda x: tokenize_light(x, vectorizer._language))
            vectorizer_.vocabulary_ = vocab

            vectorizer_._tfidf = tfidf_transformer

        vectorizer._sklearn_tfidf_vectorizer = vectorizer_
        return vectorizer


@ProcessingUnit.register("fasttext_vectorizer")
class FastTextVectorizer(ProcessingUnit):

    config_type = VectorizerConfig

    def __init__(self, config=None, **shared):
        super(FastTextVectorizer, self).__init__(config, **shared)
        self.fasttext_vectorizer = None
        self._language = None
        self.builtin_entity_scope = None

    # Simply calls "fit_transform" (to stay consistent with the "Featurizer" API)
    # We don't need "y" for the FastText vectorizer but we keep it for consistency:
    def transform(self, x):
        
        # Raw string of the input sentence:
        raw_utterance = x[0]['data'][0]['text'].strip()

        # Load the FastText model and transform the input sentence into a vector:
        fast_model = self.from_path(self.__class__.__bases__[0].by_name('language'))
        x_fasttext = fast_model[raw_utterance]

        # Convert to CSR sparse array:
        x_csr = sp.csr_matrix(x_fasttext)
        
        '''
        # Bot's model ID:
        model_id = self.__class__.__bases__[0].by_name('model_id')

        # Load the TF-IDF model and transform the input sentence into a vector:
        tfidf_model = joblib.load(f"nfs_server/SnipsModel-{model_id}/probabilistic_intent_parser/intent_classifier/tfidf.joblib")
        x_tfidf = tfidf_model.transform([raw_utterance])

        # Concatenate the two vectors:
        x_csr = sparse.hstack([x_fasttext, x_tfidf])
        '''

        return x_csr

    
    # Fit the FastText vectorizer:
    def fit_transform(self, x, y, dataset):

        # Instantiate TF-IDF:
        self.fasttext_vectorizer = FastTextVectorizer(config=self.config,
                                                builtin_entity_parser=self.builtin_entity_parser,
                                                custom_entity_parser=self.custom_entity_parser,
                                                resources=self.resources,
                                                random_state=self.random_state,
                                                )
        

        # Fit entity parsers and get resources:
        self.load_resources_if_needed(dataset[LANGUAGE])
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)
        
        # Set language:
        self._language = dataset[LANGUAGE]
        
        # Initialize FastText model:
        fast_model = self.from_path(self._language) 

        # Enrich utterances with builtin entities:
        utterances = [self._enrich_utterance(*data) for data in zip(*self._preprocess(x))]


        '''# TF-IDF embeddings:
        from sklearn.feature_extraction.text import (TfidfVectorizer as SklearnTfidfVectorizer)
        self._sklearn_tfidf_vectorizer = SklearnTfidfVectorizer()

        # Transformed data:
        x_tfidf = self._sklearn_tfidf_vectorizer.fit_transform(utterances)'''

        # Fit the FastText vectorizer => outputs dense Numpy array of arrays:
        x_fasttext = fast_model[utterances]

        # Merge FastText and TFIDF (converted to Numpy array to Scipy CSR sparse matrix):
        
        x_csr = sp.csr_matrix(x_fasttext)

        return x_csr

    
    def _preprocess(self, utterances):
        normalized_utterances = deepcopy(utterances)
        for u in normalized_utterances:
            nb_chunks = len(u[DATA])
            for i, chunk in enumerate(u[DATA]):
                chunk[TEXT] = _normalize_stem(chunk[TEXT], self.language, self.resources, self.config.vectorizer_config.use_stemming)
                if i < nb_chunks - 1:
                    chunk[TEXT] += " "

        # Extract builtin entities on unormalized utterances
        builtin_ents = [
            self.builtin_entity_parser.parse(
                get_text_from_chunks(u[DATA]),
                self.builtin_entity_scope, use_cache=True)
            for u in utterances
        ]
        # Extract builtin entities on normalized utterances
        custom_ents = [
            self.custom_entity_parser.parse(
                get_text_from_chunks(u[DATA]), use_cache=True)
            for u in normalized_utterances
        ]
        if self.config.vectorizer_config.word_clusters_name:
            # Extract world clusters on unormalized utterances
            original_utterances_text = [get_text_from_chunks(u[DATA])
                                        for u in utterances]
            w_clusters = [
                _get_word_cluster_features(
                    tokenize_light(u.lower(), self.language),
                    self.config.vectorizer_config.word_clusters_name,
                    self.resources)
                for u in original_utterances_text
            ]
        else:
            w_clusters = [None for _ in normalized_utterances]

        return normalized_utterances, builtin_ents, custom_ents, w_clusters


    # Enriches the utterance with builtin entities, custom entities and word clusters:
    def _enrich_utterance(self, utterance, builtin_entities, custom_entities,word_clusters):
        
        custom_entities_features = [_entity_name_to_feature(e[ENTITY_KIND], self.language) for e in custom_entities]

        builtin_entities_features = [_builtin_entity_to_feature(ent[ENTITY_KIND], self.language) for ent in builtin_entities]

        # We remove values of builtin slots from the utterance to avoid
        # learning specific samples such as '42' or 'tomorrow'
        filtered_tokens = [
            chunk[TEXT] for chunk in utterance[DATA]
            if ENTITY not in chunk or not is_builtin_entity(chunk[ENTITY])
        ]

        features = get_default_sep(self.language).join(filtered_tokens)

        if builtin_entities_features:
            features += " " + " ".join(sorted(builtin_entities_features))
        if custom_entities_features:
            features += " " + " ".join(sorted(custom_entities_features))
        if word_clusters:
            features += " " + " ".join(sorted(word_clusters))

        return features

    def from_path(self, lang, path=None, **shared):
        import compress_fasttext
        return compress_fasttext.models.CompressedFastTextKeyedVectors.load(f'./resources/embeddings/cc.{lang}.300-quantized')


    # Doesn't need to be persisted (pre-trained model) => pass because it's an abstract method:    
    def persist(self, path):
        pass

    # FasText is pre-trained, so it's always fitted (i.e. always True):
    def fitted(self):
        return True
    
    @property
    def language(self):
        # Create this getter to prevent the language from being set elsewhere than in the fit
        return self._language



@ProcessingUnit.register("sbert_vectorizer")
class SBERTVectorizer(ProcessingUnit):

    config_type = VectorizerConfig

    def __init__(self, config=None, **shared):
        super(SBERTVectorizer, self).__init__(config, **shared)
        self.sbert_vectorizer = None
        self._language = None
        self.builtin_entity_scope = None

    # This is used for the parse (i.e. prediction time) after the model was already trained:
    def transform(self, x):
        
        # Raw string of the input sentence:
        raw_utterance = x[0]['data'][0]['text'].strip()

        # Load the FastText model and transform the input sentence into a vector:
        sbert_model = self.from_path(self.__class__.__bases__[0].by_name('language'))
        x_sbert = sbert_model.encode(raw_utterance)

        # Convert to CSR sparse array:
        x_csr = sp.csr_matrix(x_sbert)
        
        '''
        # Bot's model ID:
        model_id = self.__class__.__bases__[0].by_name('model_id')

        # Load the TF-IDF model and transform the input sentence into a vector:
        tfidf_model = joblib.load(f"nfs_server/SnipsModel-{model_id}/probabilistic_intent_parser/intent_classifier/tfidf.joblib")
        x_tfidf = tfidf_model.transform([raw_utterance])

        # Concatenate the two vectors:
        x_csr = sparse.hstack([x_fasttext, x_tfidf])
        '''

        return x_csr

    
    # Fit the FastText vectorizer:
    def fit_transform(self, x, y, dataset):

        # Instantiate TF-IDF:
        self.sbert_vectorizer = SBERTVectorizer(config=self.config,
                                                builtin_entity_parser=self.builtin_entity_parser,
                                                custom_entity_parser=self.custom_entity_parser,
                                                resources=self.resources,
                                                random_state=self.random_state,
                                                )
        

        # Fit entity parsers and get resources:
        self.load_resources_if_needed(dataset[LANGUAGE])
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)
        
        # Set language:
        self._language = dataset[LANGUAGE]
        
        # Initialize FastText model:
        sbert_model = self.from_path(self._language) 

        # Enrich utterances with builtin entities:
        utterances = [self._enrich_utterance(*data) for data in zip(*self._preprocess(x))]

        '''# TF-IDF embeddings:
        from sklearn.feature_extraction.text import (TfidfVectorizer as SklearnTfidfVectorizer)
        self._sklearn_tfidf_vectorizer = SklearnTfidfVectorizer()

        # Transformed data:
        x_tfidf = self._sklearn_tfidf_vectorizer.fit_transform(utterances)'''

        # Fit the SBERT vectorizer => outputs dense Numpy array of arrays:
        x_sbert = sbert_model.encode(utterances)

        # Converted Numpy dense array to Scipy CSR sparse matrix:
        x_csr = sp.csr_matrix(x_sbert)

        return x_csr

    
    def _preprocess(self, utterances):
        normalized_utterances = deepcopy(utterances)
        for u in normalized_utterances:
            nb_chunks = len(u[DATA])
            for i, chunk in enumerate(u[DATA]):
                chunk[TEXT] = _normalize_stem(chunk[TEXT], self.language, self.resources, self.config.vectorizer_config.use_stemming)
                if i < nb_chunks - 1:
                    chunk[TEXT] += " "

        # Extract builtin entities on unormalized utterances
        builtin_ents = [
            self.builtin_entity_parser.parse(
                get_text_from_chunks(u[DATA]),
                self.builtin_entity_scope, use_cache=True)
            for u in utterances
        ]
        # Extract builtin entities on normalized utterances
        custom_ents = [
            self.custom_entity_parser.parse(
                get_text_from_chunks(u[DATA]), use_cache=True)
            for u in normalized_utterances
        ]
        if self.config.vectorizer_config.word_clusters_name:
            # Extract world clusters on unormalized utterances
            original_utterances_text = [get_text_from_chunks(u[DATA])
                                        for u in utterances]
            w_clusters = [
                _get_word_cluster_features(
                    tokenize_light(u.lower(), self.language),
                    self.config.vectorizer_config.word_clusters_name,
                    self.resources)
                for u in original_utterances_text
            ]
        else:
            w_clusters = [None for _ in normalized_utterances]

        return normalized_utterances, builtin_ents, custom_ents, w_clusters


    # Enriches the utterance with builtin entities, custom entities and word clusters:
    def _enrich_utterance(self, utterance, builtin_entities, custom_entities,word_clusters):
        
        custom_entities_features = [_entity_name_to_feature(e[ENTITY_KIND], self.language) for e in custom_entities]

        builtin_entities_features = [_builtin_entity_to_feature(ent[ENTITY_KIND], self.language) for ent in builtin_entities]

        # We remove values of builtin slots from the utterance to avoid
        # learning specific samples such as '42' or 'tomorrow'
        filtered_tokens = [
            chunk[TEXT] for chunk in utterance[DATA]
            if ENTITY not in chunk or not is_builtin_entity(chunk[ENTITY])
        ]

        features = get_default_sep(self.language).join(filtered_tokens)

        if builtin_entities_features:
            features += " " + " ".join(sorted(builtin_entities_features))
        if custom_entities_features:
            features += " " + " ".join(sorted(custom_entities_features))
        if word_clusters:
            features += " " + " ".join(sorted(word_clusters))

        return features

    def from_path(self, lang, path=None, **shared):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


    # Doesn't need to be persisted (pre-trained model) => pass because it's an abstract method:    
    def persist(self, path):
        pass

    # SBERT is pre-trained, so it's always fitted (i.e. always True):
    def fitted(self):
        return True
    
    @property
    def language(self):
        # Create this getter to prevent the language from being set elsewhere than in the fit
        return self._language


@ProcessingUnit.register("cooccurrence_vectorizer")
class CooccurrenceVectorizer(ProcessingUnit):
    """Featurizer that takes utterances and extracts ordered word cooccurrence
     features matrix from them"""

    config_type = CooccurrenceVectorizerConfig

    def __init__(self, config=None, **shared):
        super(CooccurrenceVectorizer, self).__init__(config, **shared)
        self._word_pairs = None
        self._language = None
        self.builtin_entity_scope = None

    @property
    def language(self):
        # Create this getter to prevent the language from being set elsewhere
        # than in the fit
        return self._language

    @property
    def word_pairs(self):
        return self._word_pairs

    def fit(self, x, dataset):
        """Fits the CooccurrenceVectorizer

        Given a list of utterances the CooccurrenceVectorizer will extract word
        pairs appearing in the same utterance. The order in which the words
        appear is kept. Additionally, if self.config.window_size is not None
        then the vectorizer will only look in a context window of
        self.config.window_size after each word.

        Args:
            x (iterable): list of utterances
            dataset (dict): dataset from which x was extracted (needed to
                extract the language and the builtin entity scope)

        Returns:
            :class:`.CooccurrenceVectorizer`: The fitted vectorizer
        """
        self.load_resources_if_needed(dataset[LANGUAGE])
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)

        self._language = dataset[LANGUAGE]

        self.builtin_entity_scope = set(e for e in dataset[ENTITIES] if is_builtin_entity(e))

        preprocessed = self._preprocess(list(x))

        utterances = [self._enrich_utterance(utterance, builtin_ents, custom_ent)
                      for utterance, builtin_ents, custom_ent in zip(*preprocessed)]
        
        # Create a set of all word pairs appearing in the utterances:
        word_pairs = set(p for u in utterances for p in self._extract_word_pairs(u))

        self._word_pairs = {pair: i for i, pair in enumerate(sorted(word_pairs))}

        return self

    @property
    def fitted(self):
        """Whether or not the vectorizer is fitted"""
        return self.word_pairs is not None

    def fit_transform(self, x, dataset):
        """Fits the vectorizer and returns the feature matrix

        Args:
            x (iterable): iterable of 3-tuples of the form
                (tokenized_utterances, builtin_entities, custom_entities)
            dataset (dict): dataset from which x was extracted (needed to
                extract the language and the builtin entity scope)

        Returns:
            :class:`.scipy.sparse.csr_matrix`: A sparse matrix X of shape
            (len(x), len(self.word_pairs)) where
            X[i, j] = 1.0 if x[i][0] contains the words cooccurrence
            (w1, w2) and if self.word_pairs[(w1, w2)] = j
        """
        return self.fit(x, dataset).transform(x)

    def _enrich_utterance(self, x, builtin_ents, custom_ents):
        utterance = get_text_from_chunks(x[DATA])
        all_entities = builtin_ents + custom_ents
        placeholder_fn = self._placeholder_fn
        # Replace entities with placeholders
        enriched_utterance = replace_entities_with_placeholders(
            utterance, all_entities, placeholder_fn)[1]
        # Tokenize
        enriched_utterance = tokenize_light(enriched_utterance, self.language)
        # Remove the unknown word strings if needed
        if self.config.unknown_words_replacement_string:
            enriched_utterance = [
                t for t in enriched_utterance
                if t != self.config.unknown_words_replacement_string
            ]
        return enriched_utterance

    #@fitted_required
    def transform(self, x):
        """Computes the cooccurrence feature matrix.

        Args:
            x (list of dict): list of utterances

        Returns:
            :class:`.scipy.sparse.csr_matrix`: A sparse matrix X of shape
            (len(x), len(self.word_pairs)) where X[i, j] = 1.0 if
            x[i][0] contains the words cooccurrence (w1, w2) and if
            self.word_pairs[(w1, w2)] = j

        Raises:
            NotTrained: when the vectorizer is not fitted
        """
        import numpy as np
        import scipy.sparse as sp

        preprocessed = self._preprocess(x)
        utterances = [
            self._enrich_utterance(utterance, builtin_ents, custom_ent)
            for utterance, builtin_ents, custom_ent in zip(*preprocessed)]
        
        if self.word_pairs == None:
            self._word_pairs = {}
            
        x_coo = sp.dok_matrix((len(x), len(self.word_pairs)), dtype=np.int32)
            
        for i, u in enumerate(utterances):
            for p in self._extract_word_pairs(u):
                if p in self.word_pairs:
                    x_coo[i, self.word_pairs[p]] = 1

        return x_coo.tocsr()

    def _preprocess(self, x):
        # Extract all entities on unnormalized data
        builtin_ents = [
            self.builtin_entity_parser.parse(
                get_text_from_chunks(u[DATA]),
                self.builtin_entity_scope,
                use_cache=True
            ) for u in x
        ]
        custom_ents = [
            self.custom_entity_parser.parse(
                get_text_from_chunks(u[DATA]), use_cache=True)
            for u in x
        ]
        return x, builtin_ents, custom_ents

    def _extract_word_pairs(self, utterance):
        if self.config.filter_stop_words:
            stop_words = get_stop_words(self.resources)
            utterance = [t for t in utterance if t not in stop_words]
        pairs = set()
        for j, w1 in enumerate(utterance):
            max_index = None
            if self.config.window_size is not None:
                max_index = j + self.config.window_size + 1
            for w2 in utterance[j + 1:max_index]:
                key = (w1, w2)
                if not self.config.keep_order:
                    key = tuple(sorted(key))
                pairs.add(key)
        return pairs

    @fitted_required
    def limit_word_pairs(self, word_pairs):
        """Restrict the vectorizer word pairs to the given word pairs

        Args:
            word_pairs (iterable of 2-tuples (str, str)): word_pairs to keep

        Returns:
            :class:`.CooccurrenceVectorizer`: The vectorizer with limited
            word pairs
        """
        word_pairs = set(word_pairs)
        existing_pairs = set(self.word_pairs)
        extra_values = word_pairs - existing_pairs

        if extra_values:
            raise ValueError(
                "Invalid word pairs %s, expected values in word_pairs"
                % sorted(extra_values))

        self._word_pairs = {
            ng: new_i for new_i, ng in enumerate(sorted(word_pairs))
        }
        return self

    def _placeholder_fn(self, entity_name):
        return "".join(
            tokenize_light(str(entity_name), str(self.language))).upper()

    @check_persisted_path
    def persist(self, path):
        path.mkdir()

        builtin_entity_scope = None
        if self.builtin_entity_scope is not None:
            builtin_entity_scope = list(self.builtin_entity_scope)

        self_as_dict = {
            "language_code": self.language,
            "word_pairs": {
                i: list(p) for p, i in iteritems(self.word_pairs)
            },
            "builtin_entity_scope": builtin_entity_scope,
            "config": self.config.to_dict()
        }
        vectorizer_json = json_string(self_as_dict)
        vectorizer_path = path / "vectorizer.json"
        with vectorizer_path.open(mode="w", encoding="utf8") as f:
            f.write(vectorizer_json)
        self.persist_metadata(path)

    @classmethod
    # pylint: disable=protected-access
    def from_path(cls, path, **shared):
        path = Path(path)
        model_path = path / "vectorizer.json"
        if not model_path.exists():
            raise LoadingError("Missing vectorizer model file: %s"
                               % model_path.name)

        with model_path.open(encoding="utf8") as f:
            vectorizer_dict = json.load(f)
        config = vectorizer_dict.pop("config")

        self = cls(config, **shared)
        self._language = vectorizer_dict["language_code"]
        self._word_pairs = None

        builtin_entity_scope = vectorizer_dict["builtin_entity_scope"]
        if builtin_entity_scope is not None:
            builtin_entity_scope = set(builtin_entity_scope)
        self.builtin_entity_scope = builtin_entity_scope

        if vectorizer_dict["word_pairs"]:
            self._word_pairs = {
                tuple(p): int(i)
                for i, p in iteritems(vectorizer_dict["word_pairs"])
            }
        return self


def VectorizerFactory(vectorizer_name, featurizer_config):

    if vectorizer_name == 'tfidf_vectorizer':
        return TfidfVectorizer(featurizer_config)
    
    elif vectorizer_name == 'fasttext_vectorizer':
        return FastTextVectorizer(featurizer_config)
    
    elif vectorizer_name == 'sbert_vectorizer':
        return SBERTVectorizer(featurizer_config)
    

def _entity_name_to_feature(entity_name, language):
    return "entityfeature%s" % "".join(tokenize_light(entity_name.lower(), language))


def _builtin_entity_to_feature(builtin_entity_label, language):
    return "builtinentityfeature%s" % "".join(tokenize_light(builtin_entity_label.lower(), language))


def _normalize_stem(text, language, resources, use_stemming):
    from snips_nlu_utils import normalize

    if use_stemming:
        return stem(text, language, resources)
    return normalize(text)


def _get_word_cluster_features(query_tokens, clusters_name, resources):
    if not clusters_name:
        return []
    ngrams = get_all_ngrams(query_tokens)
    cluster_features = []
    for ngram in ngrams:
        cluster = get_word_cluster(resources, clusters_name).get(
            ngram[NGRAM].lower(), None)
        if cluster is not None:
            cluster_features.append(cluster)
    return cluster_features
