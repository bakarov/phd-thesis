import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import List, Dict
from pandas import DataFrame


class Solver:
    def __init__(self, clf, metric, vectors_list: List):
        self.clf = clf
        self.metric = metric
        self.result_table = DataFrame(index=vectors_list)

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

    def evaluate(self, vectors_1: List, vectors_2: List, labels_: List, task: str) -> float:
        vectors = self.cosine_vectors(vectors_1, vectors_2)
        if task == 'nli':
            labels = self.encode_labels(labels_)
        elif task == 'sts':
            labels = labels_
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(vectors, labels, test_size=0.1,
                                                                                random_state=42)
        self.fit()
        self.predict()
        return self.get_result()
    
    def conduct_experiment(self, vectors_dict: Dict, labels: List, task: str, experiment_name: str):
        results = []
        for vectors_name, vectors in vectors_dict.items():
            results.append(self.evaluate(vectors[0], vectors[1], labels, task))
        self.result_table[experiment_name] = results
        
    def get_result_table(self):
        return self.result_table
