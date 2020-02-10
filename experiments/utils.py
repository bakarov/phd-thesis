from typing import List
from gensim.models import KeyedVectors
import numpy as np


def get_vector(text: List, model: KeyedVectors) -> np.ndarray:
    vector = np.zeros(shape=model.vector_size)
    for word in text:
        if word in model.vocab:
            vector = np.add(vector, model[word])
    try:
        return vector / len(text)
    except FloatingPointError:
        return vector
