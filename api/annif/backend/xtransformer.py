"""Annif backend using the transformer variant of pecos."""
# https://github.com/NatLibFi/Annif/pull/540

import logging
import os.path as osp
from sys import stdout

import numpy as np
import scipy.sparse as sp
from pecos.utils.featurization.text.preprocess import Preprocessor
from pecos.xmc.xtransformer import matcher
from pecos.xmc.xtransformer.model import XTransformer
from pecos.xmc.xtransformer.module import MLProblemWithText

from annif.exception import NotInitializedException, NotSupportedException
from annif.suggestion import SubjectSuggestion
from annif.util import (
    apply_param_parse_config,
    atomic_save,
    atomic_save_folder,
    boolean,
)

from . import backend, mixins


class XTransformerBackend(mixins.TfidfVectorizerMixin, backend.AnnifBackend):
    """XTransformer based backend for Annif"""

    name = "xtransformer"
    needs_subject_index = True

    _model = None

    train_X_file = "xtransformer-train-X.npz"
    train_y_file = "xtransformer-train-y.npz"
    train_txt_file = "xtransformer-train-raw.txt"
    model_folder = "xtransformer-model"

    model_shortcut = "distilbert-base-multilingual-cased" #"bert-base-multilingual-cased"

    PARAM_CONFIG = {
        "min_df": int, # Minimum document frequency for a term to be included in the vocab
        "ngram": int, # Considers n-amount of words in sequence, for capturing linguistic context and relationships in text.
        "fix_clustering": boolean,
        "nr_splits": int,
        "min_codes": int,
        "max_leaf_size": int,
        "imbalanced_ratio": float,
        "imbalanced_depth": int,
        "max_match_clusters": int,
        "do_fine_tune": boolean,
        "model_shortcut": str,
        "beam_size": int,
        "limit": int,
        "post_processor": str,
        "negative_sampling": str,
        "ensemble_method": str,
        "threshold": float,
        "loss_function": str,
        "truncate_length": int,
        "hidden_droput_prob": float,
        "batch_size": int,
        "gradient_accumulation_steps": int,
        "learning_rate": float,
        "weight_decay": float,
        "adam_epsilon": float,
        "num_train_epochs": int,
        "max_steps": int,
        "lr_schedule": str,
        "warmup_steps": int,
        "logging_steps": int,
        "save_steps": int,
        "max_active_matching_labels": int,
        "max_num_labels_in_gpu": int,
        "use_gpu": boolean,
        "bootstrap_model": str,
    }

    DEFAULT_PARAMETERS = {
        "min_df": 1,
        "ngram": 1,
        "fix_clustering": False,
        "nr_splits": 16,
        "min_codes": None,
        "max_leaf_size": 100,
        "imbalanced_ratio": 0.0,
        "imbalanced_depth": 100,
        "max_match_clusters": 32768,
        "do_fine_tune": True,
        "model_shortcut": model_shortcut,
        "beam_size": 20,
        "limit": 100,
        "post_processor": "sigmoid",
        "negative_sampling": "tfn",
        "ensemble_method": "transformer-only",
        "threshold": 0.1,
        "loss_function": "squared-hinge",
        "truncate_length": 128,
        "hidden_droput_prob": 0.1,
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "adam_epsilon": 1e-8,
        "num_train_epochs": 3,
        "max_steps": 0,
        "lr_schedule": "linear",
        "warmup_steps": 0,
        "logging_steps": 100,
        "save_steps": 1000,
        "max_active_matching_labels": None,
        "max_num_labels_in_gpu": 65536,
        "use_gpu": True,
        "bootstrap_model": "linear",
    }

    def _initialize_model(self):
        ''' This function checks if the model is available, if it exists it will load it, otherwise raise an exception.'''
        if self._model is None:
            path = osp.join(self.datadir, self.model_folder)
            self.debug("loading model from {}".format(path))
            if osp.exists(path):
                self._model = XTransformer.load(path)
            else:
                raise NotInitializedException(
                    "model {} not found".format(path), backend_id=self.backend_id
                )

    def initialize(self, parallel=False):
        self.initialize_vectorizer()
        self._initialize_model()

    def default_params(self):
        params = backend.AnnifBackend.DEFAULT_PARAMETERS.copy()
        params.update(self.DEFAULT_PARAMETERS)
        return params

    def _create_train_files(self, veccorpus, corpus):
        self.info("creating training file")
        # Initialize lists to store feature vectors (Xs) and labels (ys)
        Xs = []
        ys = []

        # Open a text file in write mode for storing the training data
        txt_pth = osp.join(self.datadir, self.train_txt_file) # 
        with open(txt_pth, "w", encoding="utf-8") as txt_file:
            # Iterate over pairs of documents and vectors
            for doc, vector in zip(corpus.documents, veccorpus):
                # Check if the document has a non-empty subject set and non-empty text
                subject_set = doc.subject_set
                if not (subject_set and doc.text):
                    continue 

                # Print the cleaned text to the text file
                print(" ".join(doc.text.split()), file=txt_file)

                # Convert the vector into a CSR matrix and store the sorted indices
                Xs.append(sp.csr_matrix(vector, dtype=np.float32).sorted_indices())

                # Create a label matrix (ys) with ones at positions corresponding to subjects in the document's subject set
                ys.append(
                    sp.csr_matrix(
                        (
                            np.ones(len(subject_set)),
                            (np.zeros(len(subject_set)), [s for s in subject_set]),
                        ),
                        shape=(1, len(self.project.subjects)),
                        dtype=np.float32,
                    ).sorted_indices()
                )

        # Save the stacked feature matrix (Xs) as a compressed sparse row matrix
        atomic_save(
            sp.vstack(Xs, format="csr"),
            self.datadir,
            self.train_X_file,
            method=lambda mtrx, target: sp.save_npz(target, mtrx, compressed=True),
        )

        # Save the stacked label matrix (ys) as a compressed sparse row matrix
        atomic_save(
            sp.vstack(ys, format="csr"),
            self.datadir,
            self.train_y_file,
            method=lambda mtrx, target: sp.save_npz(target, mtrx, compressed=True),
        )

    def _create_model(self, params, jobs):

        # Load training data from file
        train_txts = Preprocessor.load_data_from_file(
            osp.join(self.datadir, self.train_txt_file),
            label_text_path=None,
            text_pos=0,
        )["corpus"]

        # Load the training feature matrix (X) and label matrix (y) from saved .npz files
        train_X = sp.load_npz(osp.join(self.datadir, self.train_X_file))
        train_y = sp.load_npz(osp.join(self.datadir, self.train_y_file))

        # Set the path to save the trained model
        model_path = osp.join(self.datadir, self.model_folder)

        # Apply configuration parameters and create new parameters for training and prediction
        new_params = apply_param_parse_config(self.PARAM_CONFIG, self.params)
        new_params["only_topk"] = new_params.pop("limit")
        train_params = XTransformer.TrainParams.from_dict(
            new_params, recursive=True
        ).to_dict()
        pred_params = XTransformer.PredParams.from_dict(
            new_params, recursive=True
        ).to_dict()

        # Print an informational message indicating the start of the training process
        self.info("Start training")

        # Enable progress reporting
        matcher.LOGGER.setLevel(logging.INFO)
        matcher.LOGGER.addHandler(logging.StreamHandler(stream=stdout))

         # Train the XTransformer model using the specified parameters
        self._model = XTransformer.train(
            MLProblemWithText(train_txts, train_y, X_feat=train_X),
            clustering=None,
            val_prob=None,
            train_params=train_params,
            pred_params=pred_params,
            beam_size=params["beam_size"],
            steps_scale=None,
            label_feat=None,
        )
        # Save the trained model
        atomic_save_folder(self._model, model_path)

    def _train(self, corpus, params, jobs=0):
        # Checks if trainingdata is already cached, prints message if true
        if corpus == "cached":
            self.info("Reusing cached training data from previous run.")
        else:
            # Check if corpus is empty, raises an exception and prints message if true
            if corpus.is_empty():
                raise NotSupportedException("Cannot t project with no documents")
            
            # Prepare input for vectorization: extract text from documents
            input = (doc.text for doc in corpus.documents)

            # Define vectorization parameters based on input parameters
            vecparams = {
                "min_df": int(params["min_df"]),
                "tokenizer": self.project.analyzer.tokenize_words,
                "ngram_range": (1, int(params["ngram"])),
            }

             # Create a vectorized representation of the input corpus
            veccorpus = self.create_vectorizer(input, vecparams)

            # Create training files using the vectorized corpus and the original corpus
            self._create_train_files(veccorpus, corpus)

        # Train the model using the created training files and provided parameters
        self._create_model(params, jobs)

    def _suggest(self, text, params):
        # Clean up the input text by removing extra whitespaces
        text = " ".join(text.split())

        # Transform the cleaned text into a vector using the vectorizer
        vector = self.vectorizer.transform([text])

        # Check if the vector is an all-zero vector, indicating an empty result
        if vector.nnz == 0:
            return None
        new_params = apply_param_parse_config(self.PARAM_CONFIG, params)

        # Make predictions using the model
        prediction = self._model.predict(
            [text],
            X_feat=vector.sorted_indices(),
            batch_size=new_params["batch_size"],
            use_gpu=False,
            only_top_k=new_params["limit"],
            post_processor=new_params["post_processor"],
        )
        results = []
        for idx, score in zip(prediction.indices, prediction.data):
            results.append(SubjectSuggestion(subject_id=idx, score=score))
        return results

