import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.python.data import Dataset


dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
dataset = dataset.map(lambda x: x + 1)
print(list(dataset.as_numpy_iterator()))
