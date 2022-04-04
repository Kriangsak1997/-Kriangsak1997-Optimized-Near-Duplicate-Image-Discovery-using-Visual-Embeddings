from typing import Tuple

import numpy as np


def byte_deserializer(bts: bytes, shape: Tuple[int, int, int]) -> np.ndarray:
    # x, y, z = shape
    reads = np.frombuffer(bts)
    reshaped = reads.reshape(shape)
    reshaped = np.float32(reshaped)
    return reshaped
#
# def reshape(arr,shape):
#     x,y,x = shape
#     arr.reshape(224)
