
# SV-Softmax math


```python
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
```

Support vector guided softmax loss - is a novel loss function which adaptively emphasizes the mis-classified points (support vectors) to guide the discriminative features learning. It makes it close to hard negative mining and the Focal loss techniques.

Let's define a binary mask to adaptively indicate whether a sample is selected as the support vector by a specific classifier in the current stage. To the end, the binary mask is defined as follows:

$$
I_k = \left\{
        \begin{array}{ll}
            0, & \quad \cos(\theta_{w_y}, x) − \cos(\theta_{w_k}, x) \ge 0 \\
            1, & \quad \cos(\theta_{w_y}, x) − \cos(\theta_{w_k}, x) < 0
        \end{array}
      \right.
$$

where $\cos(\theta_{w_k}, x) = w_k^Tx$ is the cosine similarity and $θ_{w_k,x}$ is the angle between $w_k$ and $x$.


```python
# placeholders
logits = tf.placeholder(tf.float32)
y_true = tf.placeholder(tf.float32)

zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
ones = array_ops.ones_like(logits, dtype=logits.dtype)

logit_y = tf.reduce_sum(tf.multiply(y_true, logits), axis=-1, keepdims=True)
I_k = array_ops.where(logit_y >= logits, zeros, ones)

with tf.Session() as sess:
    logits_array = np.array([[2., 3., 1., -1.], [-1., 2.1, 2., 6], [-2., 3., 4, -2.1]])
    y_true_array = np.array([[0., 1., 0., 0], [0., 0., 1., 0], [1., 0., 0., 0.]])
    binary_mask = (sess.run(I_k, feed_dict={logits: logits_array, y_true: y_true_array}))
    
print("Logits:")
print(logits_array)
print('')
print("GT:")
print(y_true_array)
print('')
print("Binary mask:")
print(binary_mask)
```

    Logits:
    [[ 2.   3.   1.  -1. ]
     [-1.   2.1  2.   6. ]
     [-2.   3.   4.  -2.1]]
    
    GT:
    [[0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [1. 0. 0. 0.]]
    
    Binary mask:
    [[0. 0. 0. 0.]
     [0. 1. 0. 1.]
     [0. 1. 1. 0.]]


Let's also define indicator function $h(t, θ_{w_k}, x, I_k)$ with preset hyperparameter t:

$$h(t, θ_{w_k}, x, I_k) = e^{s(t−1)(\cos(\theta_{w_k, x})+1)I_k}$$

Obviously, when t = 1, the designed SV-Softmax loss becomes identical to the original softmax loss. Let's implement it in a naive way.


```python
# placeholders
t = tf.placeholder(tf.float32)
s = tf.placeholder(tf.float32)
logits = tf.placeholder(tf.float32)
y_true = tf.placeholder(tf.float32)
epsilon = 1.e-9

zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
ones = array_ops.ones_like(logits, dtype=logits.dtype)

logit_y = tf.reduce_sum(tf.multiply(y_true, logits), axis=-1, keepdims=True)
I_k = array_ops.where(logit_y >= logits, zeros, ones)

h = tf.exp(s * tf.multiply(t - 1., tf.multiply(logits + 1., I_k)))

# Let's check
logits_array = np.array([[2., 3., 1., -1.], [-1., 2.1, 2., 6], [-2., 3., 4, -2.1]])
y_true_array = np.array([[0., 1., 0., 0], [0., 0., 1., 0], [1., 0., 0., 0.]])
    
with tf.Session() as sess:
    h_array_12 = (sess.run(h, feed_dict={t: 1.2, s: 1, logits: logits_array, y_true: y_true_array}))
    h_array_1 = (sess.run(h, feed_dict={t: 1., s: 1, logits: logits_array, y_true: y_true_array}))
    
print("h with t=1.2:")
print(h_array_12)
print('')
print("h with t=1.0:")
print(h_array_1)
```

    h with t=1.2:
    [[1.        1.        1.        1.       ]
     [1.        1.8589282 1.        4.055201 ]
     [1.        2.2255414 2.7182825 1.       ]]
    
    h with t=1.0:
    [[1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]]


Full loss is formulated:
$$\mathcal{L} = -log\frac{e^{s\cos(\theta_{w_y}, x)}}{e^{s\cos(\theta_{w_y}, x)}+\sum_{k\ne y}^Kh(t, θ_{w_k}, x, I_k)e^{s\cos(\theta_{w_k, x})}}$$


```python
# placeholders
t = tf.placeholder(tf.float32)
s = tf.placeholder(tf.float32)
logits = tf.placeholder(tf.float32)
y_true = tf.placeholder(tf.float32)
epsilon = 1.e-9

zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
ones = array_ops.ones_like(logits, dtype=logits.dtype)

logit_y = tf.reduce_sum(tf.multiply(y_true, logits), axis=-1, keepdims=True)
I_k = array_ops.where(logit_y >= logits, zeros, ones)

h = tf.exp(s * tf.multiply(t - 1., tf.multiply(logits + 1., I_k)))


softmax = tf.exp(s * logits) / (tf.reshape(
                 tf.reduce_sum(tf.multiply(tf.exp(s * logits), h), axis=-1, keepdims=True), 
                 [-1, 1]) + epsilon)

tf_softmax = tf.nn.softmax(logits)

# Let's check softmax
logits_array = np.array([[2., 3., 1., -1.], [-1., 2.1, 2., 6], [-2., 3., 4, -2.1]])
y_true_array = np.array([[0., 1., 0., 0], [0., 0., 1., 0], [1., 0., 0., 0.]])
    
with tf.Session() as sess:
    softmax_array_12 = (sess.run(softmax, feed_dict={t: 1.2, s: 1, logits: logits_array, y_true: y_true_array}))

with tf.Session() as sess:
    softmax_array_1 = (sess.run(softmax, feed_dict={t: 1., s: 1, logits: logits_array, y_true: y_true_array}))

with tf.Session() as sess:
    tf_softmax_array = (sess.run(tf_softmax, feed_dict={t: 1., s: 1, logits: logits_array, y_true: y_true_array}))
    
print("Softmax with t=1.2:")
print(softmax_array_12)
print('')
print("Softmax with t=1.0:")
print(softmax_array_1)
print('')
print("Pure softmax:")
print(tf_softmax_array)
print('')
print("Maximum absolute error between our and tf sodtmax:")
print(abs((tf_softmax_array-softmax_array_1)).max())
print('')

# Full loss:
softmax = tf.add(softmax, epsilon)
ce = tf.multiply(y_true, -tf.log(softmax))
ce = tf.reduce_sum(ce, axis=1)
ce = tf.reduce_mean(ce)

# tf loss:
tf_ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=logits)
tf_ce = tf.reduce_mean(tf_ce)

with tf.Session() as sess:
    loss_12 = (sess.run(ce, feed_dict={t: 1.2, s: 1, logits: logits_array, y_true: y_true_array}))
    loss_1 = (sess.run(ce, feed_dict={t: 1., s: 1, logits: logits_array, y_true: y_true_array}))
    loss_tf = (sess.run(tf_ce, feed_dict={t: 1., s: 1, logits: logits_array, y_true: y_true_array}))

print("Loss with t=1.2:")
print(loss_12)
print('')
print("Loss with t=1.0:")
print(loss_1)
print('')
print("tf loss with t=1.0:")
print(loss_tf)
```

    Softmax with t=1.2:
    [[2.4178252e-01 6.5723300e-01 8.8946812e-02 1.2037643e-02]
     [2.2175812e-04 4.9225753e-03 4.4541308e-03 2.4318731e-01]
     [6.9986947e-04 1.0386984e-01 2.8234750e-01 6.3326815e-04]]
    
    Softmax with t=1.0:
    [[2.4178252e-01 6.5723300e-01 8.8946812e-02 1.2037643e-02]
     [8.7725709e-04 1.9473307e-02 1.7620180e-02 9.6202922e-01]
     [1.8058796e-03 2.6801631e-01 7.2854382e-01 1.6340276e-03]]
    
    Pure softmax:
    [[2.4178253e-01 6.5723306e-01 8.8946819e-02 1.2037644e-02]
     [8.7725703e-04 1.9473307e-02 1.7620180e-02 9.6202922e-01]
     [1.8058793e-03 2.6801628e-01 7.2854376e-01 1.6340273e-03]]
    
    Maximum absolute error between our and tf sodtmax:
    5.9604645e-08
    
    Loss with t=1.2:
    4.366085
    
    Loss with t=1.0:
    3.5917118
    
    tf loss with t=1.0:
    3.5917118


As your remember, $h(t, θ_{w_k}, x, I_k) = e^{s(t−1)(\cos(\theta_{w_k, x})+1)I_k}$, so we can rewrite our loss in this way: $$\mathcal{L} = -log\frac{e^{s\cos(\theta_{w_y}, x)}}{e^{s\cos(\theta_{w_y}, x)}+\sum_{k\ne y}^Ke^{s(t−1)(\cos(\theta_{w_k, x})+1)I_k+s\cos(\theta_{w_k, x})}}$$


```python
# placeholders
epsilon = 1.e-9
s = tf.placeholder(tf.float32)
t = tf.placeholder(tf.float32)
m = tf.placeholder(tf.float32)
logits = tf.placeholder(tf.float32)
y_true = tf.placeholder(tf.float32)

zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
ones = array_ops.ones_like(logits, dtype=logits.dtype)

# score of groundtruth
logit_y = tf.reduce_sum(tf.multiply(y_true, logits), axis=-1, keepdims=True)

# binary mask for support vectors
I_k = array_ops.where(logit_y >= logits, zeros, ones)

# indicator function
h = s * tf.multiply(t - 1., tf.multiply(logits + 1., I_k))
logits_h = tf.add(s * logits, h)
ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=logits_h)
ce = tf.reduce_mean(ce)

with tf.Session() as sess:
    loss_12 = (sess.run(ce, feed_dict={t: 1.2, s: 1, logits: logits_array, y_true: y_true_array}))
    loss_1 = (sess.run(ce, feed_dict={t: 1., s: 1, logits: logits_array, y_true: y_true_array}))

print("Loss with t=1.2:")
print(loss_12)
print('')
print("Loss with t=1.0:")
print(loss_1)
```

    Loss with t=1.2:
    4.3660855
    
    Loss with t=1.0:
    3.5917118


The same results as with our previouse code.

SV-Softmax loss semantically fuses the motivation of mining-based and margin-based losses into one framework, but from different viewpoints. Therefore, we can also absorb their strengths into our SV-Softmax loss. Specifically, to increase the mining range, we adopt the margin-based decision boundaries to indicate the support vectors. Consequently, the improved SV-X-Softmax loss can be formulated as: $$\mathcal{L} = -log\frac{e^{sf(m, \theta_{w_y}, x)}}{e^{sf(m, \theta_{w_y}, x)}+\sum_{k\ne y}^Kh(t, θ_{w_k}, x, I_k)e^{s\cos(\theta_{w_k, x})}}$$

where X is the margin-based losses. It can be A-Softmax, AM-Softmax and Arc-Softmax etc. The indicator mask I k is re-computed according to margin-based decision boundaries. Specifically:

$$
I_k = \left\{
        \begin{array}{ll}
            0, & \quad f(m, \theta_{w_y}, x) − \cos(\theta_{w_k}, x) \ge 0 \\
            1, & \quad f(m, \theta_{w_y}, x) − \cos(\theta_{w_k}, x) < 0
        \end{array}
      \right.
$$

Let's implement SV-AM-Softmax loss and done with it (for AM-Softmax we have $f(m, \theta_{w_y}, x) = \cos( \theta_{w_y}, x) − m$).


```python
# placeholders
epsilon = 1.e-9
s = tf.placeholder(tf.float32)
t = tf.placeholder(tf.float32)
m = tf.placeholder(tf.float32)
logits = tf.placeholder(tf.float32)
y_true = tf.placeholder(tf.float32)

zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
ones = array_ops.ones_like(logits, dtype=logits.dtype)

# score of groundtruth
logit_y = tf.reduce_sum(tf.multiply(y_true, logits), axis=-1, keepdims=True)

# binary mask for support vectors
I_k = array_ops.where(logit_y - m >= logits, zeros, ones)

# I_k should be zero for GT score
I_k = I_k * tf.cast(tf.not_equal(y_true, 1), tf.float32)

# indicator function
h = s * tf.multiply(t - 1., tf.multiply(logits + 1., I_k))

logits_m = tf.add(s * (logits - m * y_true), h)
ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=logits_m)
ce = tf.reduce_mean(ce)

with tf.Session() as sess:
    loss_12 = (sess.run(ce, feed_dict={m: 0.35, t: 1.2, s: 1, logits: logits_array, y_true: y_true_array}))
    loss_1 = (sess.run(ce, feed_dict={m: 0.35, t: 1., s: 1, logits: logits_array, y_true: y_true_array}))

print("Loss with t=1.2:")
print(loss_12)
print('')
print("Loss with t=1.0:")
print(loss_1)
```

    Loss with t=1.2:
    4.6436086
    
    Loss with t=1.0:
    3.8678675


---
