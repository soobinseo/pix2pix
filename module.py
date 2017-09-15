import tensorflow as tf
import numpy as np
import hyperparams as hp

def lrelu(x, leak=0.2):

    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    output = f1 * x + f2 * abs(x)
    return output


def conv2d(tensor,
           output_dim,
           filter_height=hp.filter_height,
           filter_width=hp.filter_width,
           stride=hp.stride,
           activation_fn=tf.nn.relu,
           norm_fn=tf.contrib.layers.batch_norm,
           initializer=tf.truncated_normal_initializer(stddev=0.02),
           scope="name",
           reflect=False,
           keep_prob=hp.keep_prob):

    with tf.variable_scope(scope):
        if reflect:
            tensor = tf.pad(tensor, [[0, 0], [1, 1], [1, 1], [0, 0]])
            tensor_shape = tensor.get_shape().as_list()
            filter = tf.get_variable('filter', [filter_height, filter_width, tensor_shape[-1], output_dim], initializer=initializer)
            conv = tf.nn.conv2d(tensor, filter, strides=[1, stride, stride, 1], padding='VALID')
            if norm_fn is None:
                bn = conv
            else:
                bn = tf.contrib.layers.batch_norm(conv)

            bn = tf.nn.dropout(bn, keep_prob=keep_prob)
            if activation_fn is not None:
                output = activation_fn(bn)
            else:
                output = bn

            return output
        else:
            tensor_shape = tensor.get_shape().as_list()
            filter = tf.get_variable('filters', [filter_height, filter_width, tensor_shape[-1], output_dim], initializer=initializer)
            conv = tf.nn.conv2d(tensor, filter, strides=[1, stride, stride, 1], padding='SAME')

            if norm_fn is None:
                bn = conv
            else:
                bn = tf.contrib.layers.batch_norm(conv)

            bn = tf.nn.dropout(bn, keep_prob=keep_prob)
            if activation_fn is not None:
                output = activation_fn(bn)
            else:
                output = bn

            return output

def deconv2d(tensor, output_dim, filter_height=hp.filter_height, filter_width=hp.filter_width, scope="name"): # fractional-strided conv layer

    with tf.variable_scope(scope):
        output_shape = [tf.shape(tensor)[0]] + output_dim
        filter = tf.get_variable('filters', [filter_height, filter_width, output_shape[-1], tensor.get_shape()[-1].value])
        deconv = tf.nn.conv2d_transpose(tensor, filter, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME')
        bn = tf.contrib.layers.batch_norm(tf.reshape(deconv, output_shape))
        output = tf.nn.relu(bn)

        return output


def encoder(tensor, scope="encoder"):

    with tf.variable_scope(scope):

        memory = []
        for i, out_dim in enumerate(hp.encoder_hp):
            if i == 0:
                tensor = conv2d(tensor, output_dim=out_dim, activation_fn=lrelu, norm_fn=None, scope="enc_%d" % i)
            else:
                tensor = conv2d(tensor, output_dim=out_dim, activation_fn=lrelu, scope="enc_%d" % i)
            memory.append(tensor)

        return tensor, memory

def decoder(tensor, memory, scope="decoder"):

    with tf.variable_scope(scope):
        for i, out_dim in enumerate(hp.decoder_hp):
            if i == 0:
                tensor = deconv2d(tensor, output_dim=out_dim, scope="dec_%d" % i)
            else:
                tensor = tf.concat([memory[-i-1], tensor], axis=-1)
                tensor = deconv2d(tensor, output_dim=out_dim, scope="dec_%d" % i)

        return tensor

def generator(tensor, scope="generator"):

    with tf.variable_scope(scope):
        encoded, memory = encoder(tensor)
        decoded = decoder(encoded, memory)

        return decoded

def discriminator(tensor, reuse=False, scope="discriminator"):
    with tf.variable_scope(scope, reuse=reuse):
        for i, out_dim in enumerate(hp.disc_hp):
            if i == 0:
                tensor = conv2d(tensor,
                                output_dim=out_dim,
                                activation_fn=lrelu,
                                norm_fn=None,
                                filter_height=hp.disc_filter_height,
                                filter_width=hp.disc_filter_width,
                                scope="disc_%d" % i)
            else:
                if i != len(hp.disc_hp) - 1:
                    tensor = conv2d(tensor,
                                output_dim=out_dim,
                                filter_height=hp.disc_filter_height,
                                filter_width=hp.disc_filter_width,
                                activation_fn=lrelu,
                                scope="disc_%d" % i)

                else:
                    tensor = conv2d(tensor,
                                    output_dim=out_dim,
                                    filter_height=3,
                                    filter_width=3,
                                    activation_fn=lrelu,
                                    stride=1,
                                    scope="disc_%d" % i)

        tensor = conv2d(tensor,
                        output_dim=1,
                        filter_height=3,
                        filter_width=3,
                        stride=1,
                        activation_fn=None,
                        scope="output_disc")


        return tensor

def dataset_shuffling(x, y):
    shuffled_idx = np.arange(len(y))
    np.random.shuffle(shuffled_idx)
    return x[shuffled_idx, :], y[shuffled_idx, :]

def get_batch(x, y, curr_index, batch_size):
    batch_x = x[curr_index * batch_size: (curr_index+1)*batch_size]
    batch_y = y[curr_index * batch_size: (curr_index+1)*batch_size]
    return batch_x, batch_y


