from sklearn.tree import DecisionTreeClassifier
import os.path
from typing import TYPE_CHECKING, Any
import joblib
import numpy as np
from scipy.sparse import csr_matrix 

import annif.util
from annif.exception import NotInitializedException, NotSupportedException
from annif.suggestion import SubjectSuggestion, SuggestionBatch
from annif.corpus.document import DocumentCorpus
import nltk
nltk.download('punkt')

from . import backend, mixins

if TYPE_CHECKING:
    from scipy.sparse._csr import csr_matrix

    from annif.corpus.document import DocumentCorpus

class DecisionTreeBackend(mixins.TfidfVectorizerMixin, backend.AnnifBackend):
    """Decision Tree classifier backend for Annif"""

    name = "decision_tree"

    # defaults for uninitialized instances
    _model = None

    MODEL_FILE = "decision-tree-model.gz"

    DEFAULT_PARAMETERS = {"min_df": 1, "ngram": 1}

    
    # def _initialize_model(self) -> None:
    #     print(f"Debug: Backend ID: {self.backend_id}")
    #     if self._model is None:
    #         path = os.path.join(self.datadir, self.MODEL_FILE)
    #         print(f"Debug: Loading model from {path}")
    #         if os.path.exists(path):
    #             self._model = joblib.load(path)
    #         else:
    #             raise NotInitializedException(
    #                 f"Model {path} not found", backend_id=self.backend_id
    #             )

    def _initialize_model(self) -> None:
        if self._model is None:
            path = os.path.join(self.datadir, self.MODEL_FILE)
            self.debug("loading model from {}".format(path))
            if os.path.exists(path):
                self._model = joblib.load(path)
            else:
                raise NotInitializedException(
                    "model {} not found".format(path), backend_id=self.backend_id
                )

    def initialize(self, parallel: bool = False) -> None:
        self.initialize_vectorizer()
        self._initialize_model()

    def _corpus_to_texts_and_classes(
        self, corpus: DocumentCorpus
    ) -> tuple[list[str], list[int]]:
        texts = []
        classes = []
        for doc in corpus.documents:
            if len(doc.subject_set) > 1:
                self.warning(
                    "training on a document with multiple subjects is not "
                    + "supported by Decision Tree; selecting one random subject."
                )
            elif not doc.subject_set:
                continue  # skip documents with no subjects
            texts.append(doc.text)
            classes.append(doc.subject_set[0])
        return texts, classes

    def _train_classifier(self, veccorpus: csr_matrix, classes: list[int]) -> None:
        self.info("creating decision tree classifier")
        self._model = DecisionTreeClassifier()
        self._model.fit(veccorpus, classes)
        annif.util.atomic_save(
            self._model, self.datadir, self.MODEL_FILE, method=joblib.dump
        )

    def _train(
        self, corpus: DocumentCorpus, params: dict[str, Any], jobs: int = 0
    ) -> None:
        if corpus == "cached":
            raise NotSupportedException(
                "Decision Tree backend does not support reuse of cached training data."
            )
        if corpus.is_empty():
            raise NotSupportedException("Cannot train Decision Tree project with no documents")
        texts, classes = self._corpus_to_texts_and_classes(corpus)
        vecparams = {
            "min_df": int(params["min_df"]),
            "tokenizer": self.project.analyzer.tokenize_words,
            "ngram_range": (1, int(params["ngram"])),
        }
        veccorpus = self.create_vectorizer(texts, vecparams)
        self._train_classifier(veccorpus, classes)

    def _scores_to_suggestions(
        self, scores: np.ndarray, params: dict[str, Any]
    ) -> list[SubjectSuggestion]:
        results = []
        limit = int(params["limit"])
        for class_id in np.argsort(scores)[::-1][:limit]:
            if class_id < len(self._model.classes_):
                subject_id = self._model.classes_[class_id]
                if subject_id is not None:
                    results.append(SubjectSuggestion(subject_id=subject_id, score=scores[class_id]))

        return results

    # def _suggest_batch(
    #     self, texts: list[str], params: dict[str, Any]
    # ) -> SuggestionBatch:
    #     vector = self.vectorizer.transform(texts)
    #     predictions = self._model.predict(vector)
    #     scores = self._model.predict_proba(vector)
    #     return SuggestionBatch.from_sequence(
    #         [
    #             [] if not doc.any() else
    #             [
    #                 SubjectSuggestion(subject_id=subject_id, score=scores[idx, class_id])
    #                 for class_id, subject_id in enumerate(self._model.classes_)
    #             ]
    #             for doc in predictions
    #         ],
    #         self.project.subjects,
    #     )
    
    def _suggest_batch(self, texts: list[str], params: dict[str, Any]) -> SuggestionBatch:
        vector = self.vectorizer.transform(texts)
        predictions = self._model.predict(vector)
        scores = self._model.predict_proba(vector)
        return SuggestionBatch.from_sequence(
        [
            [] if not doc.any() else
            [
                SubjectSuggestion(subject_id=subject_id, score=scores[idx, class_id])
                for class_id, subject_id in enumerate(self._model.classes_)
            ]
            for idx, doc in enumerate(predictions)
        ],
        self.project.subjects,
    )