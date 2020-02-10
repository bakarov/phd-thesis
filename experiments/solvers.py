import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import List


class Solver:
    def __init__(self, clf, metric):
        self.clf = clf
        self.metric = metric

    def concat_vectors(self, vectors_1: List, vectors_2: List) -> np.ndarray:
        return np.array(
            [np.concatenate((vector_1, vector_2), axis=0) for vector_1, vector_2 in zip(vectors_1, vectors_2)])

    def cosine_vectors(self, vectors_1: List, vectors_2: List) -> np.ndarray:
        cosines = np.array(
            [1 - distance.cosine(vector_1, vector_2) for vector_1, vector_2 in zip(vectors_1, vectors_2)]).reshape(-1,
                                                                                                                   1)
        # print(np.argwhere(np.isnan(cosines)))
        return np.nan_to_num(cosines, nan=1)

    def fit(self):
        self.clf.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.clf.predict(self.X_test)

    def encode_labels(self, labels: List):
        self.le = LabelEncoder()
        return self.le.fit_transform(labels)

    def get_result(self):
        return round(self.metric(self.y_pred, self.y_test), 3)

    def evaluate(self, vectors_1: List, vectors_2: List, labels_: List, encode_labels=False) -> float:
        vectors = self.cosine_vectors(vectors_1, vectors_2)
        if encode_labels:
            labels = self.encode_labels(labels_)
        else:
            labels = labels_
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(vectors, labels, test_size=0.1,
                                                                                random_state=42)
        self.fit()
        self.predict()
        return self.get_result()
