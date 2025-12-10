import tensorflow as tf
from tensorflow.keras import layers


class BatchNorm(object):
    def __init__(self, epsilon=1e-5, momentum=0.99, name="batch_norm"):
        self.epsilon = epsilon
        self.momentum = momentum
        self.name = name

    def __call__(self, x, is_training):
        """
        Einfacher BatchNorm-Wrapper, kompatibel mit TF1-Graphmode und bool-Placeholder `is_training`.
        """
        x_shape = x.get_shape()[-1]
        if x_shape is None:
            raise ValueError("BatchNorm input must have known channel dimension.")
        channels = int(x_shape)

        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            beta = tf.compat.v1.get_variable(
                "beta", [channels], initializer=tf.zeros_initializer()
            )
            gamma = tf.compat.v1.get_variable(
                "gamma", [channels], initializer=tf.ones_initializer()
            )
            moving_mean = tf.compat.v1.get_variable(
                "moving_mean", [channels],
                initializer=tf.zeros_initializer(),
                trainable=False,
            )
            moving_var = tf.compat.v1.get_variable(
                "moving_variance", [channels],
                initializer=tf.ones_initializer(),
                trainable=False,
            )

            # alle Achsen außer der letzten (Kanäle) normalisieren
            ndims = len(x.get_shape())
            axes = list(range(ndims - 1))

            def bn_train():
                mean, var = tf.nn.moments(x, axes=axes, keepdims=False)
                update_mean = tf.compat.v1.assign(
                    moving_mean,
                    moving_mean * self.momentum + mean * (1.0 - self.momentum),
                )
                update_var = tf.compat.v1.assign(
                    moving_var,
                    moving_var * self.momentum + var * (1.0 - self.momentum),
                )
                with tf.control_dependencies([update_mean, update_var]):
                    return tf.nn.batch_normalization(
                        x, mean, var, beta, gamma, self.epsilon
                    )

            def bn_infer():
                return tf.nn.batch_normalization(
                    x, moving_mean, moving_var, beta, gamma, self.epsilon
                )

            return tf.cond(is_training, bn_train, bn_infer)


def FE_layer(inputs, cout, aggregate_global=True, bn_is_training=True, scope="FE_layer"):
    """

    :param inputs: a tensor of shape (batch_size, num_pts, cin)
    :param cout: # out channels
    :return:  a tensor of shape (batch_size, num_pts, cout)
    """
    if aggregate_global:
        channel = cout // 2
    else:
        channel = cout
    cin = inputs.get_shape().as_list()[-1]
    with tf.compat.v1.variable_scope(scope) as local_scope:
        num_pts = inputs.get_shape().as_list()[1]

        # point-wise dense layer (Keras 3-kompatibel)
        point_wise_feature = layers.Dense(
            units=channel,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name="dense",
        )(inputs)

        batch_norm = BatchNorm()
        point_wise_feature = batch_norm(point_wise_feature, is_training=bn_is_training)
        point_wise_feature = tf.nn.leaky_relu(point_wise_feature)  # (batch_size, num_pts, cout // 2)
        if aggregate_global:
            aggregated_feature = tf.reduce_max(input_tensor=point_wise_feature, axis=1, keepdims=True)
            repeated = tf.tile(aggregated_feature, [1, num_pts, 1])
            point_wise_concatenated_feature = tf.concat(axis=-1, values=[point_wise_feature, repeated])
            return point_wise_feature, point_wise_concatenated_feature
        else:
            return point_wise_feature, point_wise_feature


def dense_norm_nonlinear(inputs, units,
                         norm_type=None,
                         is_training=None,
                         activation_fn=tf.nn.relu,
                         scope="fc"):
    """
    :param inputs: tensor of shape (batch_size, ...,n) from last layer
    :param units: output units
    :param norm_type: a string indicating which type of normalization is used.
                    A string start with "b": use batch norm.
                    A string starting with "l": use layer norm
                    others: do not use normalization
    :param is_training: a boolean placeholder of shape () indicating whether its in training phase or test phase.
    It is only needed when BN is used.
    :param activation_fn:
    :param scope: scope name
    :return: (batch_size, ...,units)
    """
    with tf.compat.v1.variable_scope(scope):
        # dichte Schicht (Keras 3-kompatibel)
        out = layers.Dense(
            units=units,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name="dense",
        )(inputs)

        if norm_type is not None:
            if norm_type.lower().startswith("b"):
                batch_norm = BatchNorm()
                if is_training is None:
                    raise ValueError("is_training is not given!")
                out = batch_norm(out, is_training=is_training)
            elif norm_type.lower().startswith("l"):
                out = tf.contrib.layers.layer_norm(out, scope="layer_norm")
            elif norm_type.lower().startswith("i"):
                out = tf.contrib.layers.instance_norm(out, scope="instance_norm")
            else:
                raise ValueError("please give the right norm type beginning with 'b' or 'l'!")
        if activation_fn is not None:
            out = activation_fn(out)
        return out
