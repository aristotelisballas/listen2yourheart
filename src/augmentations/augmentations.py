import numpy as np
import tensorflow as tf
from scipy.signal import buttord, butter, filtfilt
from tensorflow import Tensor
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow_io.python.ops.audio_ops import resample

from utilities.signalutils import analogue2digital


# np_config.enable_numpy_behavior()


class CutOffFilter:
    def __init__(self, f_pass_Hz: float, f_stop_Hz: float, fs_Hz: float, *, f_rand_offset_Hz: float = 0.0):
        self._pass: float = f_pass_Hz
        self._stop: float = f_stop_Hz
        self._nyquist: float = fs_Hz / 2
        self._rand_offset: float = f_rand_offset_Hz

    def __str__(self) -> str:
        return f'CutOffFilter: f_pass={self._pass} Hz, f_stop={self._stop} Hz, fs={2 * self._nyquist} Hz, f_rand_offset={self._rand_offset} Hz'

    def _apply_filter(self, x: np.ndarray) -> np.ndarray:
        f_rand_offset = tf.random.uniform([1], -self._rand_offset, self._rand_offset)
        n, wn = buttord(
            wp=(self._pass + f_rand_offset) / self._nyquist,
            ws=(self._stop + f_rand_offset) / self._nyquist,
            gpass=3,
            gstop=30
        )

        btype: str
        if self._pass > self._stop:
            btype = 'high'
        else:
            btype = 'low'

        sos = butter(n, wn, btype)
        b = sos[0]
        a = sos[1]

        return filtfilt(b, a, x).astype(np.half)

    @tf.function(input_signature=[tf.TensorSpec(None, tf.half)])
    def __call__(self, x: Tensor):
        return tf.numpy_function(self._apply_filter, [x], tf.half)


class FlipLR:
    def __init__(self, p: float = 0.5):
        self._p: float = p

    def __str__(self) -> str:
        return f'FlipLR: p={100 * self._p}%'

    def __call__(self, x):
        return tf.map_fn(
            fn=lambda x: tf.reverse(x, axis=[0]) if tf.random.uniform([1, ], 0, 1) < self._p else x, elems=x
        )


class FlipRandom:
    def __init__(self, p_lr: float = 0.5, p_ud: float = 0.5):
        self._lr: FlipLR = FlipLR(p=p_lr)
        self._ud: FlipUD = FlipUD(p=p_ud)

    def __str__(self) -> str:
        return f'FlipRandom: p_lr={100 * self._lr._p}%, p_ud={100 * self._ud._p}%'

    def __call__(self, x: Tensor):
        return self._lr(self._ud(x))


class FlipUD:
    def __init__(self, p: float = 0.5):
        self._p: float = p

    def __str__(self) -> str:
        return f'FlipUD: p={100 * self._p}%'

    def __call__(self, x):
        return tf.map_fn(fn=lambda x: -x if tf.random.uniform([1, ], 0, 1) < self._p else x, elems=x)


class NoAugmentation():
    def __call__(self, x):
        return x

    def __str__(self) -> str:
        return 'NoAugmentation'


class RandomResample:
    def __init__(self, fs_hz: float, p: float = 0.5, a_min: float = 0.5, a_max: float = 2):
        self._fs: float = fs_hz
        self._p: float = p
        self._a_min: float = a_min
        self._a_max: float = a_max

    def __str__(self) -> str:
        return f'RandomResample: fs={self._fs} Hz, p={100 * self._p}%, a in [{self._a_min}, {self._a_max}]'

    def fn(self, x: Tensor):
        if tf.random.uniform([1, ], 0.0, 1.0) >= self._p:
            return x
        else:
            rate_in = tf.cast(tf.round(self._fs), dtype=tf.int64)
            a = tf.random.uniform([1, ], self._a_min, self._a_max)
            rate_out = tf.cast(tf.round(a * self._fs), dtype=tf.int64)
            y = resample(input=tf.cast(x, dtype=tf.float32), rate_in=rate_in, rate_out=rate_out)
            y = tf.cast(y, dtype=tf.float16)
            len_x = tf.shape(x)
            len_y = tf.shape(y)
            if a < 1.0:
                shape = len_x - len_y
                shape = tf.where(tf.equal(shape, 0), 1, shape)
                padding = tf.zeros(shape, dtype=y.dtype)
                z = tf.concat((y, padding), axis=0)
            else:
                m = tf.cast(tf.round((len_y - len_x) / 2), dtype=tf.int32)
                z = tf.slice(y, m, len_x)
            return z

    def __call__(self, x: Tensor):
        return tf.map_fn(fn=self.fn, elems=x)


class RandomScaling:
    def __init__(self, a_min: float = 0.5, a_max: float = 0.5):
        self._a_min: float = a_min
        self._a_max: float = a_max

    def __str__(self) -> str:
        return f'RandomScaling: a in [{self._a_min}, {self._a_max}]'

    def __call__(self, x):
        return tf.map_fn(
            fn=lambda x: tf.random.uniform([1], self._a_min, self._a_max, dtype=tf.half) * x, elems=x
        )


class Trim:
    def __init__(self, length_sec: float, max_delay_sec: float, fs_hz: float):
        self._length_sec: float = length_sec
        self._max_delay_sec: float = max_delay_sec
        self._fs: float = fs_hz

        self.slice_size: Tensor = tf.constant([analogue2digital(length_sec, fs_hz), ], tf.int32)
        self.max_delay: int = analogue2digital(max_delay_sec, fs_hz)

    def __str__(self) -> str:
        return f'Trim: length={self._length_sec} sec, max_delay={self._max_delay_sec} sec, fs={self._fs} Hz'

    def __call__(self, x):
        begin: Tensor = tf.random.uniform(shape=[1, ], minval=0, maxval=self.max_delay, dtype=tf.int32)

        return tf.slice(input=x, begin=begin, size=self.slice_size)


class UniformNoise:
    def __init__(self, v_min: float, v_max: float):
        self._v_min: float = v_min
        self._v_max: float = v_max

    def __str__(self) -> str:
        return f'UniformNoise: v in [{self._v_min}, {self._v_max}]'

    def __call__(self, x):
        v = tf.random.uniform(x.shape, self._v_min, self._v_max, dtype=tf.half)

        return x + v
