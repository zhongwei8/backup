# Copyright 2020 Xiaomi
import onnxruntime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (LSTM, BatchNormalization, Concatenate,
                                     Conv1D, Conv2D, Convolution1D, Dense,
                                     Dropout, Flatten, GlobalAveragePooling1D,
                                     GlobalAveragePooling2D, Input,
                                     MaxPooling1D, MaxPooling2D, Multiply,
                                     Permute)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow_model_optimization as tfmot

from . import tcresnet_blocks


class ONNXPredictor(object):
    def __init__(self, onnx_model_file) -> None:
        self.session = onnxruntime.InferenceSession(onnx_model_file)

    def predict(self, data):
        inputs = {self.session.get_inputs()[0].name: data}
        return self.session.run(None, inputs)


class GHM_Loss:
    def __init__(self, bins=10, momentum=0.75):
        self.g = None
        self.bins = bins
        self.momentum = momentum
        self.valid_bins = tf.constant(0.0, dtype=tf.float32)
        self.edges_left, self.edges_right = self.get_edges(self.bins)
        if momentum > 0:
            acc_sum = [0.0 for _ in range(bins)]
            self.acc_sum = tf.Variable(acc_sum, trainable=False)

    @staticmethod
    def get_edges(bins):
        edges_left = [float(x) / bins for x in range(bins)]
        edges_left = tf.constant(edges_left)  # [bins]
        edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1]
        edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1, 1]
        edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1, 1, 1]

        edges_right = [float(x) / bins for x in range(1, bins + 1)]
        edges_right[-1] += 1e-3
        edges_right = tf.constant(edges_right)  # [bins]
        edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1]
        edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1, 1]
        edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1, 1, 1]
        return edges_left, edges_right

    def calc(self, g, valid_mask):
        edges_left, edges_right = self.edges_left, self.edges_right
        alpha = self.momentum
        # valid_mask = tf.cast(valid_mask, dtype=tf.bool)

        tot = tf.maximum(tf.reduce_sum(tf.cast(valid_mask, dtype=tf.float32)),
                         1.0)
        inds_mask = tf.logical_and(tf.greater_equal(g, edges_left),
                                   tf.less(g, edges_right))
        zero_matrix = tf.cast(tf.zeros_like(inds_mask),
                              dtype=tf.float32)  # [bins, batch_num, class_num]

        inds = tf.cast(tf.logical_and(inds_mask, valid_mask),
                       dtype=tf.float32)  # [bins, batch_num, class_num]

        num_in_bin = tf.reduce_sum(inds, axis=[1, 2, 3])  # [bins]
        valid_bins = tf.greater(num_in_bin, 0)  # [bins]

        num_valid_bin = tf.reduce_sum(tf.cast(valid_bins, dtype=tf.float32))

        if alpha > 0:
            update = tf.compat.v1.assign(
                self.acc_sum,
                tf.where(valid_bins,
                         alpha * self.acc_sum + (1 - alpha) * num_in_bin,
                         self.acc_sum))
            with tf.control_dependencies([update]):
                acc_sum_tmp = tf.identity(self.acc_sum, name='updated_accsum')
                acc_sum = tf.expand_dims(acc_sum_tmp, -1)  # [bins, 1]
                acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1]
                acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1, 1]
                acc_sum = acc_sum + zero_matrix  # [bins, batch_num, class_num]
                weights = tf.where(tf.equal(inds, 1), tot / acc_sum,
                                   zero_matrix)
                weights = tf.reduce_sum(weights, axis=0)

        else:
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1]
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1]
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1, 1]
            num_in_bin = num_in_bin + zero_matrix  # [bins, batch_num, class_num]
            weights = tf.where(tf.equal(inds, 1), tot / num_in_bin,
                               zero_matrix)
            weights = tf.reduce_sum(weights, axis=0)

        weights = weights / num_valid_bin

        return weights, tot

    def ghm_class_loss(self, logits, targets, masks=None):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        """
        train_mask = (1 - tf.cast(tf.equal(targets, -1), dtype=tf.float32))
        self.g = tf.abs(tf.sigmoid(logits) - targets)  # [batch_num, class_num]
        g = tf.expand_dims(self.g, axis=0)  # [1, batch_num, class_num]

        if masks is None:
            masks = tf.ones_like(targets)
        valid_mask = masks > 0
        weights, tot = self.calc(g, valid_mask)
        print(weights.shape)
        ghm_class_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=targets * train_mask, logits=logits)
        ghm_class_loss = tf.reduce_sum(ghm_class_loss * weights) / tot

        return ghm_class_loss

    def ghm_regression_loss(self, logits, targets, masks):
        """ Args:
        input [batch_num, *(* class_num)]:
            The prediction of box regression layer. Channel number can be 4 or
            (4 * class_num) depending on whether it is class-agnostic.
        target [batch_num,  *(* class_num)]:
            The target regression values with the same size of input.
        """
        mu = self.mu

        # ASL1 loss
        diff = logits - targets
        # gradient length
        g = tf.abs(diff / tf.sqrt(mu * mu + diff * diff))

        if masks is None:
            masks = tf.ones_like(targets)
        valid_mask = masks > 0

        weights, tot = self.calc(g, valid_mask)

        ghm_reg_loss = tf.sqrt(diff * diff + mu * mu) - mu
        ghm_reg_loss = tf.reduce_sum(ghm_reg_loss * weights) / tot

        return ghm_reg_loss


ghm = GHM_Loss(bins=30)


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

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)

    return focal_loss_fixed


def cnn_model_base(n_timesteps,
                   n_features,
                   n_outputs,
                   hidden_layers=[8, 16, 32, 16],
                   kernels=[5, 5, 5, 5],
                   pool_size=2,
                   dropout_rate=0.5,
                   l2_penalty=0.05,
                   learning_rate=0.0003,
                   activation='relu'):
    """
    Keras Multi-layer neural network. Fixed parameters include:
    1. activation function (PRelu)
    2. always uses batch normalization after the activation
    3. use adam as the optimizer

    Parameters
    ----------
    Tunable parameters are (commonly tuned)

    hidden_layers: list
        the number of hidden layers, and the size of each hidden layer

    kernels: list
        the kernel size for each hidden layer

    dropout_rate: float 0 ~ 1
        if bigger than 0, there will be a dropout layer

    l2_penalty: float
        or so called l2 regularization

    optimizer: string or keras optimizer
        method to train the network

    Returns
    -------
    model :
        a keras model

    Reference
    ---------
    https://keras.io/scikit-learn-api/
    """
    model = Sequential()
    for index, layers in enumerate(hidden_layers):
        if not index:
            model.add(
                Conv1D(layers,
                       kernels[index],
                       input_shape=(n_timesteps, n_features),
                       padding='same',
                       activation=activation))
            model.add(
                MaxPooling1D(pool_size=pool_size, padding='same', strides=2))
            model.add(BatchNormalization())
        else:
            model.add(
                Conv1D(layers,
                       kernels[index],
                       padding='same',
                       activation=activation))
            model.add(
                MaxPooling1D(pool_size=pool_size, padding='same', strides=2))

    model.add(GlobalAveragePooling1D())
    model.add(Dropout(dropout_rate))
    model.add(
        Dense(n_outputs,
              activation='softmax',
              kernel_regularizer=regularizers.l2(l2_penalty)))
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss=focal_loss(alpha=1),
                  metrics=['categorical_accuracy'])
    return model


def cnn_model(n_timesteps, n_features, n_outputs):
    model = Sequential()
    model.add(
        Conv2D(12,
               kernel_size=(1, 5),
               input_shape=(1, n_timesteps, n_features),
               padding='same',
               activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2), padding='same', strides=2))
    model.add(Conv2D(12, kernel_size=(1, 5), padding='same',
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2), padding='same', strides=2))
    model.add(Conv2D(16, kernel_size=(1, 5), padding='same',
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(1, 2), padding='same', strides=2))
    model.add(Conv2D(20, kernel_size=(1, 5), padding='same',
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2), strides=2, padding='same'))
    # model.add(LSTM(16))
    # model.add(Flatten())
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())

    model.add(
        Dense(n_outputs,
              activation='softmax',
              kernel_regularizer=regularizers.l2(0.2)))
    model.compile(optimizer=Adam(lr=0.0003),
                  loss=ghm.ghm_class_loss,
                  metrics=['categorical_accuracy'])
    return model


def cnn_model_3(n_timesteps, n_features, n_outputs):
    model = Sequential()
    model.add(
        Conv1D(8,
               5,
               input_shape=(n_timesteps, n_features),
               padding='same',
               activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding='same', strides=2))
    model.add(BatchNormalization())
    model.add(Conv1D(16, 5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding='same', strides=2))
    # model.add(Conv1D(32, 5, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=2, padding='same', strides=2))
    model.add(Conv1D(16, 5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    # model.add(LSTM(16))
    # model.add(Flatten())
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(
        Dense(n_outputs,
              activation='softmax',
              kernel_regularizer=regularizers.l2(0.05)))
    model.compile(optimizer=Adam(lr=0.0003),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def cnn_features_model(n_timesteps, n_features, n_outputs, nb_features):
    input = Input(shape=(n_timesteps, n_features))
    conv_1 = Convolution1D(12,
                           5,
                           input_shape=(n_timesteps, n_features),
                           padding='same',
                           activation='relu')(input)
    bat_1 = BatchNormalization()(conv_1)
    # drop_1 = Dropout(0.3)(bat_1)
    maxp_1 = MaxPooling1D(pool_size=2, padding='same', strides=2)(bat_1)

    conv_2 = Convolution1D(12,
                           5,
                           input_shape=(n_timesteps, n_features),
                           padding='same',
                           activation='relu')(maxp_1)
    bat_2 = BatchNormalization()(conv_2)
    # drop_2 = Dropout(0.3)(conv_2)
    maxp_2 = MaxPooling1D(pool_size=2, padding='same', strides=2)(bat_2)

    conv_3 = Convolution1D(20,
                           5,
                           input_shape=(n_timesteps, n_features),
                           padding='same',
                           activation='relu')(maxp_2)
    bat_3 = BatchNormalization()(conv_3)
    drop_3 = Dropout(0.3)(bat_3)
    maxp_3 = MaxPooling1D(pool_size=2, padding='same', strides=2)(drop_3)

    conv_4 = Convolution1D(16,
                           5,
                           input_shape=(n_timesteps, n_features),
                           padding='same',
                           activation='relu')(maxp_3)
    bat_4 = BatchNormalization()(conv_4)
    # drop_4 = Dropout(0.3)(conv_4)
    maxp_4 = MaxPooling1D(pool_size=2, padding='same', strides=2)(bat_4)

    # drop = Dropout(0.3)(maxp_4)
    seq_features = GlobalAveragePooling1D()(maxp_4)

    other_features = Input(shape=(nb_features, ))
    model = Concatenate()([seq_features, other_features])
    # model.add(Flatten())
    model = Dense(n_outputs,
                  activation='softmax',
                  kernel_regularizer=regularizers.l2(0.2))(model)
    model = Model([input, other_features], model)
    model.compile(
        optimizer=Adam(lr=0.0003),
        # loss='categorical_crossentropy',
        # loss=focal_loss(gamma=2,alpha=1),
        loss=ghm.ghm_class_loss,
        metrics=['categorical_accuracy'])
    return model


def lstm_model(n_timesteps, n_features, n_outputs, hidden_size, dense_size):
    model = Sequential()
    model.add(
        LSTM(hidden_size,
             return_sequences=False,
             unroll=False,
             input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(dense_size, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])
    return model


def ann_model(n_features, n_outputs):
    model = Sequential()
    model.add(Dense(64, input_dim=n_features, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(196, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(optimizer=Adam(lr=0.0003),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def attention_3d_block(inputs, n):
    print(inputs)
    a = Permute((2, 1))(inputs)
    # print(a)
    a = Dense(n, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    # print(a_probs)
    output = Multiply(name='attention_mul')([inputs, a_probs])
    return output


def autoencoder(n_timesteps, n_features):

    input_img = Input(shape=(n_timesteps * n_features, ))
    encoded = Dense(512, activation='relu')(input_img)
    encoded = Dense(256, activation='relu')(encoded)
    encoder_output = Dense(128, activation='relu')(encoded)

    decoded = Dense(128, activation='relu')(encoder_output)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = Dense(n_timesteps * n_features, activation='relu')(decoded)

    autoencoder = Model(inputs=input_img, outputs=decoded)
    encoder = Model(inputs=input_img, outputs=encoder_output)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder


def tc_resnet8_model(n_timesteps, n_features, n_outputs):

    blocks = [2, 2, 2]
    classes = n_outputs
    width_multiplier = 1
    block = tcresnet_blocks.basic_2d
    k = width_multiplier
    features = [8 * k, 12 * k, 16 * k]
    input = Input(shape=(n_timesteps, 1, n_features))
    x = keras.layers.Conv2D(4 * k, (5, 1),
                            strides=(1, 1),
                            use_bias=False,
                            name="conv1",
                            padding="same",
                            kernel_regularizer=regularizers.l2(0.002))(input)
    x = keras.layers.BatchNormalization(name="bn_conv1")(x)
    x = keras.layers.Activation("relu", name="conv1_relu")(x)

    for stage_id, stride in enumerate(blocks):
        x = block(
            features[stage_id],
            stage_id,
            stride,
            numerical_name=True,
        )(x)

    x = keras.layers.GlobalAveragePooling2D(name="pool5")(x)
    x = keras.layers.Dense(classes, activation="softmax", name="fc2")(x)
    x = Model(input, x)
    x.compile(optimizer=Adam(lr=0.0003),
              loss=ghm.ghm_class_loss,
              metrics=['categorical_accuracy'])
    return x


quantize_model = tfmot.quantization.keras.quantize_model


def q_tc_resnet_model(n_timesteps, n_features, n_outputs):
    blocks = [2, 2, 2]
    classes = n_outputs
    width_multiplier = 1
    block = tcresnet_blocks.basic_2d
    k = width_multiplier
    features = [8 * k, 12 * k, 16 * k]
    input = Input(shape=(n_timesteps, 1, n_features))
    x = keras.layers.Conv2D(4 * k, (5, 1),
                            strides=(1, 1),
                            use_bias=False,
                            name="conv1",
                            padding="same",
                            kernel_regularizer=regularizers.l2(0.002))(input)
    x = keras.layers.BatchNormalization(name="bn_conv1")(x)
    x = keras.layers.Activation("relu", name="conv1_relu")(x)

    for stage_id, stride in enumerate(blocks):
        x = block(
            features[stage_id],
            stage_id,
            stride,
            numerical_name=True,
        )(x)

    x = keras.layers.GlobalAveragePooling2D(name="pool5")(x)
    x = keras.layers.Dense(classes, activation="softmax", name="fc2")(x)
    x = Model(input, x)
    x = quantize_model(x)
    x.compile(optimizer=Adam(lr=0.0003),
              loss=ghm.ghm_class_loss,
              metrics=['categorical_accuracy'])
    return x
