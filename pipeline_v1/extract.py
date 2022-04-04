import numpy as np
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
from keras.applications.resnet import ResNet50
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


# Extract feature embeddings
def extract(array: np.ndarray, batch_size: int) -> np.ndarray:
    features = model.predict(array, batch_size=batch_size)
    return features
