import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from utils.activation import swish
from utils.loss import ncce


class Model:
    def __init__(self, type, vocabulary_size, embedding_dim, embedding_matrix, trainable, params, filename='',
                 optimizer=tf.keras.optimizers.Adagrad(), loss=ncce):
        self.type = type
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.trainable = trainable
        self.params = params
        self.sequence_length = 100
        self.n_out = 2
        self.optimizer = optimizer
        self.loss = loss
        self.filename = filename

    def fit(self, x_train, y_train, x_test, y_test, batch_size, epochs, i):
        self.sequence_length = x_train.shape[1]
        self.n_out = y_train.shape[1]
        if self.type == 'LSTM':
            model = self.lstm()
        else:
            model = self.cnn()

        checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(self.filename, 'weights.best.{}.hdf5'.format(i)),
                                                        monitor='val_loss', verbose=1,
                                                        save_best_only=True, mode='min')

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[tf.keras.metrics.Precision(name='precision'),
                      tf.keras.metrics.Recall(name='recall')])
        #print(model.summary())
        learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                       patience=2,
                                                                       verbose=1,
                                                                       factor=0.5,
                                                                       min_lr=0.00001)
        earlystopping = tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, shuffle=True,
                  callbacks=[checkpoint, learning_rate_reduction, earlystopping],
                  validation_data=(x_test, y_test))
        return tf.keras.models.load_model(os.path.join(self.filename, 'weights.best.{}.hdf5'.format(i)),
                                           custom_objects={'swish': swish, 'ncce': ncce})



    def lstm(self):
        """
        Create LSTM model
        :return: Tensorflow model
        """

        inputs = tf.keras.Input(shape=(self.sequence_length,), dtype='int32')
        if self.embedding_matrix is not None:
            embedding = tf.keras.layers.Embedding(input_dim=self.vocabulary_size, output_dim=self.embedding_dim,
                                                  input_length=self.sequence_length,
                                                  weights=[self.embedding_matrix], trainable=self.trainable)
        else:

            embedding = tf.keras.layers.Embedding(input_dim=self.vocabulary_size, output_dim=self.embedding_dim,
                                                  input_length=self.sequence_length, trainable=True)
        embedded_sequences = embedding(inputs)

        if len(self.params['units_sizes']) > 0:
            text_features1 = tf.keras.layers.LSTM(self.params['units_size1'],
                                                  input_shape=(self.sequence_length, self.embedding_dim),
                                                  return_sequences=True)(embedded_sequences)
            i = 0
            for usz in self.params['units_sizes']:
                if i == len(self.params['units_sizes']) - 1:
                    text_features1 = tf.keras.layers.LSTM(usz, return_sequences=False)(text_features1)
                else:
                    text_features1 = tf.keras.layers.LSTM(usz, return_sequences=True)(text_features1)
                i += 1
        else:
            text_features1 = tf.keras.layers.LSTM(self.params['units_size1'],
                                                  input_shape=(self.sequence_length, self.embedding_dim),
                                                  return_sequences=False)(embedded_sequences)

        dense_sizes = self.params['dense_size2']
        dense1 = tf.keras.layers.Dense(self.params['dense_size1'], activation=swish)(text_features1)
        dense1 = tf.keras.layers.BatchNormalization()(dense1)
        for dsz in dense_sizes:
            dense1 = tf.keras.layers.Dense(dsz, activation=swish)(dense1)
            dense1 = tf.keras.layers.BatchNormalization()(dense1)
        if self.n_out == 1:
            output = tf.keras.layers.Dense(units=self.n_out, activation='sigmoid',
                                           kernel_regularizer=tf.keras.regularizers.l2(self.params['dropout']))(dense1)
        else:
            output = tf.keras.layers.Dense(units=self.n_out, activation='softmax',
                                           kernel_regularizer=tf.keras.regularizers.l2(self.params['dropout']))(dense1)
        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model

    def cnn(self):
        """
        Create CNN model
        :return: Tensorflow model
        """
        inputs = tf.keras.Input(shape=(self.sequence_length,), dtype='int32')
        if self.embedding_matrix is not None:
            embedding = tf.keras.layers.Embedding(input_dim=self.vocabulary_size, output_dim=self.embedding_dim,
                                                  input_length=self.sequence_length,
                                                  weights=[self.embedding_matrix], trainable=self.trainable)
        else:

            embedding = tf.keras.layers.Embedding(input_dim=self.vocabulary_size, output_dim=self.embedding_dim,
                                                  input_length=self.sequence_length, trainable=True)
        embedded_sequences = embedding(inputs)

        convs = []
        for fsz in self.params['filter_sizes']:
            conv = tf.keras.layers.Conv1D(filters=self.params['nb_filter'],
                                          kernel_size=fsz,
                                          padding='valid',
                                          activation='relu',
                                          strides=1)(embedded_sequences)
            pool = tf.keras.layers.MaxPooling1D(pool_size=self.sequence_length-fsz+1)(conv)
            flattenmax = tf.keras.layers.Flatten()(pool)
            convs.append(flattenmax)

        l_merge = tf.keras.layers.concatenate(convs, axis=1)
        dense_sizes = self.params['dense_size2']
        dense1 = tf.keras.layers.Dense(self.params['dense_size1'], activation=swish)(l_merge)
        dense1 = tf.keras.layers.BatchNormalization()(dense1)
        for dsz in dense_sizes:
            dense1 = tf.keras.layers.Dense(dsz, activation=swish)(dense1)
            dense1 = tf.keras.layers.BatchNormalization()(dense1)
        if self.n_out == 1:
            output = tf.keras.layers.Dense(units=self.n_out, activation='sigmoid',
                                           kernel_regularizer=tf.keras.regularizers.l2(self.params['dropout']), )(
                dense1)
        else:
            output = tf.keras.layers.Dense(units=self.n_out, activation='softmax',
                                           kernel_regularizer=tf.keras.regularizers.l2(self.params['dropout']), )(
                dense1)
        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model

