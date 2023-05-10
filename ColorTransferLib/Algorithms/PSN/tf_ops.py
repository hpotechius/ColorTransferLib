import tensorflow as tf
import tf_slim as slim


class BatchNorm(object):
    def __init__(self, epsilon=1e-5, momentum=0.99, name="batch_norm"):
        with tf.compat.v1.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, is_training):
        return slim.layers.batch_norm(x,
                                            decay=self.momentum,
                                            epsilon=self.epsilon,
                                            updates_collections=None,
                                            scale=True,
                                            is_training=is_training,
                                            scope=self.name)


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

        #point_wise_feature = tf.keras.layers.Dense(units=channel, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))(inputs)
        point_wise_feature = tf.compat.v1.layers.dense(inputs, channel,
                                                       kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                           stddev=0.02))


        batch_norm = BatchNorm()
        point_wise_feature = batch_norm(point_wise_feature, is_training=bn_is_training)
        point_wise_feature = tf.nn.leaky_relu(point_wise_feature)  # (batch_size, num_pts, cout // 2)
        if aggregate_global:
            aggregated_feature = tf.reduce_max(input_tensor=point_wise_feature, axis=1, keepdims=True)  # batch_size, 1, cout//2
            repeated = tf.tile(aggregated_feature, [1, num_pts, 1])  # (batch_size, num_pts, cout // 2)
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
        # out = tf.keras.layers.Dense(units=units,
        #                                          kernel_initializer=tf.compat.v1.truncated_normal_initializer(
        #                                              stddev=0.02))(inputs)
        out = tf.compat.v1.layers.dense(inputs, units,
                              kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
        #  batch_size, num_point, num_features
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
