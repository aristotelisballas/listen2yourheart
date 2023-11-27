import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.losses import categorical_crossentropy

LARGE_NUM = 1e9


def _loss0(labels, logits_aa, logits_ab, logits_ba, logits_bb):
    # Approach 0: SimCLR authors

    loss_a = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_ab, logits_aa], 1))
    loss_b = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_ba, logits_bb], 1))
    loss = loss_a + loss_b

    return loss


def _loss1(labels, logits_aa, logits_ab, logits_ba, logits_bb):
    # Approach 1: keras based
    loss_a = categorical_crossentropy(labels, tf.concat([logits_ab, logits_aa], 1), True)
    loss_b = categorical_crossentropy(labels, tf.concat([logits_ba, logits_bb], 1), True)
    a1_loss = loss_a + loss_b

    return a1_loss


def _loss2(labels, logits_aa, logits_ab, logits_ba, logits_bb):
    # Approach 2: analytical
    batch_size = tf.shape(logits_aa)[0]
    eye = tf.one_hot(tf.range(2 * batch_size), 2 * batch_size)

    logits = tf.concat([tf.concat([logits_ab, logits_aa], 1), tf.concat([logits_bb, logits_ba], 1)], 0)
    exp_logits = tf.math.exp(logits)
    l0 = tf.divide(tf.math.reduce_sum(eye * exp_logits, 0), tf.math.reduce_sum(exp_logits, 0))
    l_ij = -tf.math.log(l0)
    loss = tf.math.reduce_sum(tf.split(l_ij, 2, 0), 0)

    return loss


def __le_contrastive_loss(hidden, hidden_norm: bool = True, temperature: float = 1.0):
    """
    Alternative implementation of ``__contrastive_loss``. Hopefully more intuitive. Also faster.
    """
    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)

    n: int = tf.shape(hidden)[0]

    # A diagonal matrix mask, that has a LARGE_NUM value. This matrix will be subtracted from the similarity matrix in
    # order to force 0-values for the exponents, and thus eliminate the need for an indicator function
    diag_mask: Tensor = tf.math.scalar_mul(LARGE_NUM, tf.linalg.eye(n))  # this can be cached

    labels: Tensor = tf.signal.fftshift(tf.linalg.eye(n), 0)
    similarities: Tensor = tf.linalg.matmul(hidden, hidden, transpose_b=True) / temperature - diag_mask
    vloss: Tensor = categorical_crossentropy(labels, similarities, True)
    loss: Tensor = tf.math.reduce_mean(vloss)

    return loss


def __l2_normalization(tensor2d: Tensor):
    l2_norms = tf.norm(tensor2d, 2, -1)
    # squared_sum = tf.math.reduce_sum(tf.math.pow(tensor2d, 2), -1)
    # l2_norms = tf.math.sqrt(squared_sum)

    l2_norms = tf.expand_dims(l2_norms, -1)

    return tf.math.divide(tensor2d, l2_norms)


def __contrastive_loss(hidden, hidden_norm: bool = True, temperature: float = 1.0, weights: float = 1.0):
    """
    Mostly a copy-paste from: https://github.com/google-research/simclr/blob/master/objective.py#L34

    Notes on original method:
    - hidden: Tensor, shape is (1024, 128) where 1024 are the training samples (512 "original" batch size),
              and 128 is the feature vector. It seems that hidden[0:512, :] are the 512 training samples
              from the first augmentation, and hidden[512:1024, :] are the 512 training samples from the
              second augmentation
    """
    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)
        # hidden = __l2_normalization(hidden)

    # normed1 = tf.math.l2_normalize(hidden, -1)
    # normed2 = __l2_normalization(hidden)
    #
    # tf.print('norms 1', tf.norm(normed1, 2, 1))
    # tf.print('norms 2', tf.norm(normed2, 2, 1))

    hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]

    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(hidden1, hidden1, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(hidden2, hidden2, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True) / temperature
    logits_ba = tf.matmul(hidden2, hidden1, transpose_b=True) / temperature
    # NOTE logits_ba is actually logits_ab.transpose() but we compute it as per the original implementation

    # logits = tf.concat((tf.concat((logits_aa, logits_ab), 1), tf.concat((logits_ba, logits_bb), 1)), 0)
    # tf.print('\n', logits)
    # tf.print('logits aa\n', logits_aa)
    # tf.print('logits ab\n', logits_ab)
    # tf.print('logits ba\n', logits_ba)
    # tf.print('logits bb\n', logits_bb)

    loss_a = categorical_crossentropy(labels, tf.concat([logits_ab, logits_aa], 1), True)
    loss_b = categorical_crossentropy(labels, tf.concat([logits_ba, logits_bb], 1), True)
    loss = (loss_a + loss_b) / 2
    loss = tf.math.reduce_mean(loss)

    return loss, logits_ab, labels


def create_contrastive_loss(hidden_norm: bool = True, temperature: float = 1.0, weights: float = 1.0):
    """
    Creates a loss function based on Keras definition: https://keras.io/api/losses/#creating-custom-losses

    The loss function has the following signature:
        ``(y_true_unused: Tensor, y_pred: Tensor) -> Tensor``

    Note that the ground-truth labels, ``y_true_unused``, are not used at all (hence "unused"), ``y_pred`` should be
    logits, and a single scalar is returned as the loss.

    :param hidden_norm: Whether to apply L2 normalization to ``y_pred``
    :param temperature:
    :param weights:
    :return: A loss function
    """
    if weights != 1.0:
        return NotImplementedError("Weights is not implemented yet")

    def cl(_y_true: Tensor, y_pred: Tensor) -> Tensor:
        """Note that the ground-truth labels, ``_y_true``, are not used at all in this function."""
        loss, logits_ab, labels = __contrastive_loss(y_pred, hidden_norm, temperature, weights)
        # loss = __le_contrastive_loss(y_pred, hidden_norm, temperature)

        return loss

    return cl
