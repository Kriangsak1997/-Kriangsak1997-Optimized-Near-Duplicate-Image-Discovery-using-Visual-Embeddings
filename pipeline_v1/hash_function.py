from typing import List

import numpy as np

# import saved random vectors
random_vectors = np.loadtxt("random_vectors.txt")  # drawn from standard normal distribution


# hash using random projections
def hash_func(embedding: np.ndarray) -> int:
    # Random projection.
    bools = np.dot(embedding, random_vectors) > 0
    return bool2int(bools)


# packing
def bool2int(x: List[bool]) -> int:
    y = 0
    for i, j in enumerate(x):
        if j:
            y += 1 << i
    return y
