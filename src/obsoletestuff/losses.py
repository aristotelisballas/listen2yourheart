import numpy as np
import tensorflow as tf

from tensorflow import Tensor


def _get_costs() -> Tensor:
    c_algorithm = 1
    c_gp = 250
    c_specialist = 500
    c_treatment = 1000
    c_error = 10000
    a = 0.5

    c_pp = c_gp + c_treatment
    c_pu = c_gp + c_specialist + a * c_treatment
    c_pn = c_gp
    c_up = c_specialist + c_treatment
    c_uu = c_specialist + a * c_treatment
    c_un = c_specialist
    c_np = c_error
    c_nu = a * c_error
    c_nn = 0
    costs: np.ndarray = np.array([
        [c_pp, c_pu, c_pn],
        [c_up, c_uu, c_un],
        [c_np, c_nu, c_nn]
    ]) + c_algorithm

    return tf.constant(costs, tf.float32)


def challenge_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    y_true = tf.cast(y_true, tf.float32)

    costs: Tensor = _get_costs()

    a: Tensor = tf.matmul(y_pred, costs)
    b: Tensor = tf.matmul(a, tf.transpose(y_true))
    loss = tf.reduce_mean(b)

    return loss


def challenge_loss_crisp(y_true: Tensor, y_pred: Tensor) -> Tensor:
    y_true = tf.argmax(y_true)
    y_true = tf.one_hot(y_true, 3)

    return challenge_loss(y_true, y_pred)
