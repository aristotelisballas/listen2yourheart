import tensorflow as tf
from keras.optimizers.schedules.learning_rate_schedule import CosineDecay, LearningRateSchedule

from tensorflow import Tensor


class LeWarmUpAndCosineDecay(LearningRateSchedule):
    def __init__(self, base_learning_rate: float, epochs: int, batches_per_epoch: int, warmup_epochs: int,
                 warmup_scaling: str, alpha: float = 0.0, initial_epoch: int = 0):
        super().__init__()

        self.base_learning_rate: float = base_learning_rate
        self.epochs: int = epochs
        self.batches_per_epoch: int = batches_per_epoch
        self.warmup_epochs: int = warmup_epochs
        self.warmup_scaling: str = warmup_scaling

        self.total_steps: int = epochs * batches_per_epoch
        self.warmup_steps: int = warmup_epochs * batches_per_epoch

        self._cosine_decay: CosineDecay = CosineDecay(base_learning_rate, self.total_steps - self.warmup_steps, alpha)

        self.initial_epoch: int = initial_epoch

    def __call__(self, relative_step: Tensor):
        step = (self.initial_epoch * self.batches_per_epoch) + relative_step
        warmup = tf.cast(step, tf.float32) / float(self.warmup_steps)
        if self.warmup_scaling == 'linear':
            warmup = warmup * self.base_learning_rate
        elif self.warmup_scaling == 'sqrt':
            warmup = tf.sqrt(warmup) * self.base_learning_rate
        elif self.warmup_scaling == 'exp':
            warmup = (tf.exp(warmup) - 1) / (tf.exp(tf.constant(1, dtype=tf.float32)) - 1) * self.base_learning_rate

        return tf.where(step < self.warmup_steps, warmup, self._cosine_decay(step - self.warmup_steps))
