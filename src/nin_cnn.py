import tensorflow as tf
import numpy as np
from math import ceil


class TextNIN:
    """ A CNN for NLP

    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.

    Args:
        sequence_length: number of words in a a sentences.
        param num_classes: TODO
        vocab_size: number of words in the vocabulary. This will define
            the embedding layer.
        embedding_size: number of dimensions of the distributed embedding vector.
        filter_sizes: number of words that the convolutional filter should cover.
        num_filters: number of filters for each filter size

    """

    def batch_norm(self, conv, filter_size, center=True, scale=True,
                   scope=None, reuse=False):
        with tf.variable_scope('{}-bn-{}'.format(scope, filter_size)) as bnscope:
            if center and reuse:
                beta = tf.get_variable("beta", conv.get_shape()[-1],
                                       initializer=tf.constant_initializer(0.0))
            if scale and reuse:
                gamma = tf.get_variable("gamma", conv.get_shape()[-1],
                                        initializer=tf.constant_initializer(1.0))
            if reuse:
                moving_avg = tf.get_variable("moving_mean", conv.get_shape()[-1],
                                             initializer=tf.constant_initializer(0.0),
                                             trainable=False)
                moving_var = tf.get_variable("moving_variance", conv.get_shape()[-1],
                                             initializer=tf.constant_initializer(1.0),
                                             trainable=False)
            else:
                bnscope = None

            conv_bn = tf.contrib.layers.batch_norm(conv,
                                                   is_training=self.is_training,
                                                   center=center,
                                                   scale=scale,
                                                   trainable=True,
                                                   updates_collections=None,
                                                   reuse=reuse,
                                                   scope=bnscope,
                                                   decay=0.9)

            if reuse:
                bnscope.reuse_variables()
        return conv_bn

    def layer_norm(self, conv):
        conv_bn = tf.contrib.layers.layer_norm(conv)
        return conv_bn

    def init_normalization(self, conv, filter_size, out_height, scope=None):
        print('---> Normalization')
        # Add Batch Normalization
        if self.add_layer_norm:
            conv_norm = self.layer_norm(conv)
        elif self.add_bn:
            # conv_norm = tf.contrib.layers.batch_norm(conv,
            #                                        is_training=self.is_training,
            #                                        center=True,
            #                                        scale=False,
            #                                        trainable=True,
            #                                        updates_collections=None,
            #                                        scope=None,
            #                                        decay=0.9)
            conv_norm = self.batch_norm(conv, filter_size,
                                        center=self.batch_beta, scale=self.batch_gamma,
                                        scope=scope, reuse=self.batch_reuse)

        # Build the Gaussian Noise vector
        print(conv_norm)
        conv_shape = [1, out_height, 1, 1]
        print(conv_shape)
        gaussian_noise = tf.truncated_normal(conv_shape, mean=0.0, stddev=0.3,
                                             dtype=tf.float32, seed=None, name=None)

        # Return the output of the tensor of the convolution after the normalization
        # TODO: Develop the possibility of adding noise.
        # LayerNorm
        if self.add_layer_norm:
            print('---> LAYER NORM', conv_norm)
            return conv_norm

        # Gaussian Noise AND Batch Normalization
        if self.add_bn and self.add_noise:
            conv_normalize = tf.cond(self.is_training,
                                     lambda: tf.add(conv_norm, gaussian_noise),
                                     lambda: conv_norm)
            print('---> BN + NOISE', conv_normalize)
            return conv_normalize

        # Batch Normalization WITHOUT Gaussian Noise
        if self.add_bn and not self.add_noise:
            print('---> BN', conv_norm)
            return conv_norm

        # Gaussian Noise WITHOUT Batch Normalization
        if not self.add_bn and self.add_noise:
            conv_normalize = tf.cond(self.is_training,
                                     lambda: tf.add(conv, gaussian_noise),
                                     lambda: conv)
            print('---> GN', conv_normalize)
            return conv_normalize

        # Return the output of the convolution without normalization
        return conv

    def init_conv(self, conv_input, in_height,
                  kernel_height, kernel_width, channels, kernels_num,
                  scope_name, pooling=None, cnn_padding='VALID'):

        print(scope_name)
        with tf.variable_scope(scope_name) as conv_sp:
            # Define the convolution layer
            filter_shape = [kernel_height, kernel_width, channels, kernels_num]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf. Variable(tf.constant(0.1, shape=[kernels_num]), name="b")
            strides = [1, 1, 1, 1]
            if cnn_padding == 'SAME':
                # HOW TO PAD THE INPUT WHEN SAME IS ACTIVATED

                # out_height = ceil(float(in_height) / float(strides[1]))
                # out_width = ceil(float(in_width) / float(strides[2]))
                # pad_along_height = ((out_height - 1) * strides[1] +
                #                     filter_height - in_height)
                # pad_along_width = ((out_width - 1) * strides[2] +
                #                    filter_width - in_width)
                # pad_top = pad_along_height / 2
                # pad_left = pad_along_width / 2

                out_height = ceil(float(in_height) / float(strides[1]))
                pad_along_height = ((out_height - 1) * strides[1] +
                                    (kernel_height - in_height))
                pad_top = ceil(pad_along_height / 2)
                pad_bottom = pad_along_height - pad_top
                padding_range = tf.constant([[0, 0], [pad_top, pad_bottom],
                                             [0, 0], [0, 0]])
                print('----> PADDING ', padding_range, pad_along_height, pad_top, pad_bottom)
                conv_input = tf.pad(conv_input, padding_range, mode="CONSTANT")
                print('----> INPUT PAD', conv_input)
                cnn_padding = 'VALID'

            else:
                # conv_input = self.embedded_chars_expanded
                out_height = ceil(float(in_height - kernel_height + 1) / float(strides[1]))
                print(in_height, kernel_height, strides[1])

            # out_height = ceil(float(in_height - kernel_height + 1) / float(strides[1]))
            conv = tf.nn.conv2d(
                conv_input,
                W,
                strides=strides,
                # VALID -> Narrow convolution
                #       -> Output of shape [1, sequence_length - filter_size + 1, 1, 1]
                # padding = SAME or VALID
                padding=cnn_padding,
                name="conv")
            print('---> CONV: ', conv)

            print('HEIGHT', out_height, conv, conv_sp.name)


            # Add Normalization
            if self.add_noise or self.add_bn or self.add_layer_norm:
                conv = self.init_normalization(conv, kernel_height, out_height, conv_sp.name)

            # Non linear activation
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            print('---> RELU: ', h)

            if pooling == 'max_pooling':
                # Maxpooling over the outputs
                # The output will be a tensor of size [batch_size, 1, 1, num_filters]
                conv_heigh = h.get_shape()[1]
                pool_size = [1, out_height, 1, 1]
                print('pool', pool_size, out_height)
                pooled = tf.nn.max_pool(
                    h,
                    ksize=pool_size,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                print('---> MAXPOOL: ', pooled)
                return pooled
            return h

    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, kernel_sizes, num_kernels, l2_reg_lambda=0.0,
                 norm='bn', batch_reuse=False, batch_beta=True, batch_gamma=True,
                 gauss_noise=False, dropout=True):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, [], name='is_training')
        self.sequence_length = sequence_length
        self.kernels_num = num_kernels

        # Normalization parameters
        self.add_bn = True if norm == 'bn' else False
        self.add_layer_norm = True if norm == 'layer' else False
        self.batch_reuse = batch_reuse
        self.batch_beta = batch_beta
        self.batch_gamma = batch_gamma
        self.add_noise = gauss_noise


        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # 1ST LAYER: Embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            # Lookup table: W is the embedding matrix that we learn during training.
            # We initialize it using a random uniform distribution (tf.random_uniform)
            self.W_embedding = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                trainable=True,
                name="W_embedding")

            # Embedding operation, a 3D tensor ->
            # [None (batch), sequence_length, embedding_size].
            self.embedded_chars = tf.nn.embedding_lookup(self.W_embedding, self.input_x)
            # Manually add a new dimension because TensorFlow’s convolutional
            # conv2d operation expects a 4-dimensional tensor and our embedding
            # matrix doesn't have channels (different views of your input data)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)


        # 2ND LAYER: FEATURE MAPS
        # Create filters and then create the layer obtained after applying a convolution
        # and the activation function
        pooled_outputs = []
        for i, kernel_height in enumerate(kernel_sizes):
            # 5. NIN
            # print('\n---------------------------------- FST {} --------------------------------'.format(kernel_height))
            # conv_fst = self.init_conv(self.embedded_chars_expanded, self.sequence_length,
            #                           kernel_height, embedding_size, 1, self.kernels_num,
            #                           scope_name="conv_1-{}".format(kernel_height))
            #
            # print('\n---------------------------------- SND {} ------------------------------'.format(kernel_height))
            # self.add_bn, self.add_noise = False, False
            # in_height = int(conv_fst.get_shape()[1])
            # print(in_height, type(in_height),
            #       conv_fst.get_shape()[1], type(conv_fst.get_shape()[1]),
            #       int(conv_fst.get_shape()[1]), type(int(conv_fst.get_shape()[1])))
            #
            # conv_output = self.init_conv(conv_fst, in_height,
            #                              1, 1, 100, self.kernels_num,
            #                              scope_name="conv_2-{}".format(kernel_height),
            #                              pooling='max_pooling')
            # print()
            # print()
            # pooled_outputs.append(conv_output)
            # self.add_bn = True if norm == 'bn' else False
            # self.add_noise = gauss_noise

            # 6. Only 1 CNN SAME
            conv_output = self.init_conv(self.embedded_chars_expanded, self.sequence_length,
                                         kernel_height, embedding_size, 1, self.kernels_num,
                                         scope_name="conv-{}".format(kernel_height),
                                         pooling='max_pooling', cnn_padding='SAME')
            pooled_outputs.append(conv_output)
            print()
            print()

            # 7. SAME & NIN
            # print('\n---------------------------------- FST {} --------------------------------'.format(kernel_height))
            # conv_fst = self.init_conv(self.embedded_chars_expanded, self.sequence_length,
            #                           kernel_height, embedding_size, 1, self.kernels_num,
            #                           scope_name="conv_1-{}".format(kernel_height),
            #                           cnn_padding='SAME')
            #
            # print('\n---------------------------------- SND {} ------------------------------'.format(kernel_height))
            # self.add_bn, self.add_noise = False, False
            # in_height = int(conv_fst.get_shape()[1])
            # print(in_height, type(in_height),
            #       conv_fst.get_shape()[1], type(conv_fst.get_shape()[1]),
            #       int(conv_fst.get_shape()[1]), type(int(conv_fst.get_shape()[1])))
            #
            # conv_output = self.init_conv(conv_fst, in_height,
            #                              1, 1, 100, self.kernels_num,
            #                              scope_name="conv_2-{}".format(kernel_height),
            #                              pooling='max_pooling')
            # print()
            # print()
            # pooled_outputs.append(conv_output)
            # self.add_bn = True if norm == 'bn' else False
            # self.add_noise = gauss_noise

            # 8. SAME & NIN no pooling
            # print('\n---------------------------------- FST {} --------------------------------'.format(kernel_height))
            # conv_fst = self.init_conv(self.embedded_chars_expanded, self.sequence_length,
            #                           kernel_height, embedding_size, 1, self.kernels_num,
            #                           scope_name="conv_1-{}".format(kernel_height),
            #                           cnn_padding='SAME')
            #
            # print('\n---------------------------------- SND {} ------------------------------'.format(kernel_height))
            # self.add_bn, self.add_noise = False, False
            # in_height = int(conv_fst.get_shape()[1])
            # print(in_height, type(in_height),
            #       conv_fst.get_shape()[1], type(conv_fst.get_shape()[1]),
            #       int(conv_fst.get_shape()[1]), type(int(conv_fst.get_shape()[1])))
            #
            # conv_output = self.init_conv(conv_fst, in_height,
            #                              1, 1, 100, 1,
            #                              scope_name="conv_2-{}".format(kernel_height))
            # print()
            # print()
            # pooled_outputs.append(conv_output)
            # self.add_bn = True if norm == 'bn' else False
            # self.add_noise = gauss_noise


        # 3RD LAYER: CONCAT ALL THE POOLED FEATURES
        # Combine all the pooled features into one long feature vector
        # of shape [batch_size, num_filters_total]
        num_filters_total = self.kernels_num * len(kernel_sizes)
        self.h_pool = tf.concat(len(kernel_sizes), pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print('---> CONCAT:', self.h_pool)

        # 4TH LAYER: DROPOUT AND PREDICTION
        # Add dropout
        if dropout:
            print('---> DROPOUT')
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        else:
            self.h_drop = self.h_pool_flat

        # COMPUTE THE ERROR
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            print('---> W output:', W)
            print('---> b output:', b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            print('---> SCORES:', self.scores)

            # Compute the output = Wx + b
            # self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # self.predictions: Class predicted for the input
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            # if batch_norm:
            #     losses = tf.nn.softmax_cross_entropy_with_logits(self.scores_BN, self.input_y)
            # else:
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            print(self.predictions,  self.input_y)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
