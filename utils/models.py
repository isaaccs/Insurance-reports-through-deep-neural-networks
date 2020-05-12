import tensorflow as tf
from utils.activation import swish

def core_model_LSTM(sequence_input
               ,sequence_length 
               ,vocabulary_size 
               ,n_out
                ,embedding_dim
                ,embedding_matrix

                ,params={'units_size1': 256, 'units_sizes': [128], 'dense_size1': 500, 'dense_size2': [300], 'dropout': 0.004}) :

    embedding =  tf.keras.layers.Embedding(input_dim=vocabulary_size, output_dim=embedding_dim
                           , input_length=sequence_length,trainable=True)
    embedded_sequences = embedding(sequence_input)
 
   
    #####LSTM#####
    if len(params['units_sizes'])>0:
        text_features1 = tf.keras.layers.LSTM(params['units_size1'], input_shape=(sequence_length, embedding_dim), return_sequences=True)(embedded_sequences)
        i=0
        for usz in params['units_sizes']:
            if i==len(params['units_sizes'])-1:
                print(usz)
                print(params['units_sizes'][-1])
                text_features1 = tf.keras.layers.LSTM(usz, return_sequences=False)(text_features1)
            else : text_features1 = tf.keras.layers.LSTM(usz, return_sequences=True)(text_features1)
            i+=1
    else :
        text_features1 = tf.keras.layers.LSTM(params['units_size1'], input_shape=(sequence_length, embedding_dim), return_sequences=False)(embedded_sequences)
 
    dense_sizes = params['dense_size2']
    dense1= tf.keras.layers.Dense(params['dense_size1'],activation=swish)(text_features1)
    dense1=tf.keras.layers.BatchNormalization()(dense1)
    for dsz in dense_sizes:
        dense1= tf.keras.layers.Dense(dsz,activation=swish)(dense1)
        dense1=tf.keras.layers.BatchNormalization()(dense1)
   
    

    if n_out==1:
        output = tf.keras.layers.Dense(units=n_out, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(params['dropout']),)(dense1)
    else:
        output = tf.keras.layers.Dense(units=n_out, activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(params['dropout']),)(dense1)
    return output
	
	
def core_model_CNN(sequence_input
               ,sequence_length 
               ,vocabulary_size 
               ,n_out
                ,embedding_dim
                ,embedding_matrix,

                params={'filter_sizes': [1, 2], 'nb_filter': 1024, 'dense_size1': 400, 'dense_size2': [250], 'dropout': 0.001}
                ) :

 
 
    embedding =  tf.keras.layers.Embedding(input_dim=vocabulary_size, output_dim=embedding_dim
                           , input_length=sequence_length,trainable=True)
    embedded_sequences = embedding(sequence_input)

    convs = []
    for fsz in params['filter_sizes']:
        conv = tf.keras.layers.Conv1D(nb_filter=params['nb_filter'],
                         filter_length=fsz,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)(embedded_sequences)
        pool = tf.keras.layers.MaxPooling1D(pool_length=sequence_length-fsz+1)(conv)
        flattenMax = tf.keras.layers.Flatten()(pool)
        convs.append(flattenMax)
 
    l_merge = tf.keras.layers.concatenate(convs, axis=1)
    dense_sizes = params['dense_size2']
    dense1= tf.keras.layers.Dense(params['dense_size1'],activation=swish)(l_merge)
    dense1=tf.keras.layers.BatchNormalization()(dense1)
    for dsz in dense_sizes:
        dense1= tf.keras.layers.Dense(dsz,activation=swish)(dense1)
        dense1=tf.keras.layers.BatchNormalization()(dense1)
 
 
    if n_out==1:
        output = tf.keras.layers.Dense(units=n_out, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(params['dropout']),)(dense1)
    else:
        output = tf.keras.layers.Dense(units=n_out, activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(params['dropout']),)(dense1)
 
        
    return output	