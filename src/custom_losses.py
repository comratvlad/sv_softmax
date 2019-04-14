import tensorflow as tf
from tensorflow.python.ops import array_ops


def sv_softmax_loss(t=1.0, s=1):

    t = float(t)
    s = float(s)
    
    def sv_softmax_loss_fixed(y_true, logits):
        """SV-Softmax loss
        Notice: y_pred is raw logits
        Support Vector Guided Softmax Loss for Face Recognition
        https://arxiv.org/pdf/1812.11317.pdf

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
        ones = array_ops.ones_like(logits, dtype=logits.dtype)
        
        logit_y = tf.reduce_sum(tf.multiply(y_true, logits), axis=-1, keepdims=True)
        I_k = array_ops.where(logit_y >= logits, zeros, ones)
        
        h = tf.exp(s * tf.multiply(t - 1., tf.multiply(logits + 1., I_k)))
        
        softmax = tf.exp(s * logits) / (tf.reshape(
                         tf.reduce_sum(tf.multiply(tf.exp(s * logits), h), axis=-1, keepdims=True), 
                         [-1, 1]) + epsilon)
        
        # We add epsilon because log(0) = nan
        softmax = tf.add(softmax, epsilon)
        ce = tf.multiply(y_true, -tf.log(softmax))
        ce = tf.reduce_sum(ce, axis=1)
        return tf.reduce_mean(ce)
    
    return sv_softmax_loss_fixed


def sv_am_softmax_loss(t=1.0, s=1, m=0.35):

    t = float(t)
    s = float(s)
    m = float(m)
    
    def sv_am_softmax_loss_fixed(y_true, logits):
        """SV-AM-Softmax loss for multi-classification
        Notice: y_pred is raw logits
        Support Vector Guided Softmax Loss for Face Recognition
        https://arxiv.org/pdf/1812.11317.pdf

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
        ones = array_ops.ones_like(logits, dtype=logits.dtype)
        
        logit_y = tf.reduce_sum(tf.multiply(y_true, logits), axis=-1, keepdims=True)
        I_k = array_ops.where(logit_y - m >= logits, zeros, ones)
        
        # I_k should be zero for GT score
        I_k = I_k * tf.cast(tf.not_equal(y_true, 1), tf.float32)

        h = s * tf.multiply(t - 1., tf.multiply(logits + 1., I_k))
        logits_m = tf.add(s * (logits - m * y_true), h)
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=logits_m)
        return tf.reduce_mean(ce)
    
    return sv_am_softmax_loss_fixed


def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        y_pred = tf.nn.softmax(y_pred)
        
        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed
