from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, MaxPooling1D)
from keras import layers

def simple_rnn_model(input_dim, output_dim=29):
    """ 
    Build a simply recurrent network for speech
    
    Params:
        input_length (int): Length of the input sequence.
        output_dim: output dimensions of the GRU

    Returns:
        returns the RNN acoustic model
    
    Code Attribution:
        This function contains code that was updated and leveraged from the 
        Udacity Natural Language Processing Nano Degree Training material.
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())

    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ 
    Build a recurrent network for speech with batch normalization and time distributed dense layer
    
    Params:
        input_length (int): Length of the input sequence.
        unit: output dimensions of the GRU
        activation: GRU activation function
        output_dim: output dimensions of the dense connected layers

    Returns:
        returns the RNN acoustic model
    
    Code Attribution:
        This function contains code that was updated and leveraged from the 
        Udacity Natural Language Processing Nano Degree Training material.
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)

    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name="bn_rnn")(simp_rnn)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())

    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ 
    Build a recurrent network + convolutional network for speech
    
    Params:
        input_length (int): Length of the input sequence.
        filters: (int): Width of the convolution kernel.
        kernel_size: length of the 1D convolution window
        conv_stride: (int): Stride size used in 1D convolution.
        conv_border_mode: (str): Only support `same` or `valid`.
        units: output dimensions of the GRU
        output_dim: output dimensions of the dense connected layers     

    Returns:
        returns the RNN acoustic model

    Code Attribution:
        This function contains code that was updated and leveraged from the 
        Udacity Natural Language Processing Nano Degree Training material.
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)

    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)

    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)

    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name="bn_rnn")(simp_rnn)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
        
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
        
    Code Attribution:
        This is Udacity code.  No changes made to this function.        
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """
    Build a deep recurrent network for speech
    
    Params:
        input_dim (int): Length of the input sequence.
        unit: output dimensions of the GRU
        recur_layers: number of recurrent GRU layers
        output_dim: output dimensions of the dense connected layers

    Returns:
        returns the RNN acoustic model

    Code Attribution:
        This function contains code that was updated and leveraged from the 
        Udacity Natural Language Processing Nano Degree Training material.
    """
    activation = "relu"
    
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    # TODO: Add recurrent layers, each with batch normalization 
    # create rnn dictionaries
    rnn_dict = {}
    bn_rnn_dict = {}

    for layer in range(1, recur_layers+1):
        # first RNN combined with input data
        if layer == 1:
            rnn_dict["rnn_" + str(layer)] = GRU(units, 
                                                activation=activation, 
                                                return_sequences=True, 
                                                implementation=2, 
                                                name='rnn_' + str(layer))(input_data)
        
        # subsequent RNN layers combined with batch normalization
        else:
            rnn_dict["rnn_" + str(layer)] = GRU(units, 
                                    activation=activation, 
                                    return_sequences=True, 
                                    implementation=2, 
                                    name='rnn_' + str(layer))(bn_rnn_dict['bn_rnn_' + str(layer-1)])
        bn_rnn_dict["bn_rnn_" + str(layer)] = BatchNormalization(name="bn_rnn_" + str(layer))(rnn_dict['rnn_' + str(layer)])
            
    
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn_dict['bn_rnn_' + str(recur_layers)])

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ 
    Build a bidirectional recurrent network for speech
    
    Params:
        input_dim (int): Length of the input sequence.
        units: output dimensions of the GRU
        output_dim: output dimensions of the dense connected layers

    Returns:
        returns the RNN acoustic model

    Code Attribution:
        This function contains code that was updated and leveraged from the 
        Udacity Natural Language Processing Nano Degree Training material.
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2, name="bidir_rnn"))(input_data)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    
    return model

def cnn_rnn_drop_out(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """
    Build a recurrent network + convolutional + drop out network for speech
    
    Params:
        input_length (int): Length of the input sequence.
        filters: (int): Width of the convolution kernel.
        kernel_size: length of the 1D convolution window
        conv_stride: (int): Stride size used in 1D convolution.
        conv_border_mode: (str): Only support `same` or `valid`.
        units: output dimensions of the GRU
        output_dim: output dimensions of the dense connected layers     

    Returns:
        returns the RNN acoustic model 
    
    Code Attribution:
        This function contains code that was updated and leveraged from the 
        Udacity Natural Language Processing Nano Degree Training material.
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # TODO: Specify the layers in your network
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    
    # add drop out
    drop_out_1 = layers.Dropout(.2)(conv_1d)
    
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(drop_out_1)

    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)

    # add drop out
    drop_out_2 = layers.Dropout(.2)(simp_rnn)
    
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name="bn_rnn")(drop_out_2)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    
    return model

def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """    
    Build a deep recurrent network + convolutional + drop out + bidirectional network for speech
    
    Params:
        input_length (int): Length of the input sequence.
        filters: (int): Width of the convolution kernel.
        kernel_size: length of the 1D convolution window
        conv_stride: (int): Stride size used in 1D convolution.
        conv_border_mode: (str): Only support `same` or `valid`.
        units: output dimensions of the GRU
        output_dim: output dimensions of the dense connected layers     

    Returns:
        returns the RNN acoustic model
    
    Code Attribution:
        This function contains code that was updated and leveraged from the 
        Udacity Natural Language Processing Nano Degree Training material.    
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # TODO: Specify the layers in your network
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    
    # add drop out
    drop_out_1 = layers.Dropout(.2)(conv_1d)
    
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(drop_out_1)

    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)

    # add drop out
    drop_out_2 = layers.Dropout(.2)(simp_rnn)
    
    # Add batch normalization
    bn_rnn = BatchNormalization(name="bn_rnn")(drop_out_2)

    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    
    # Add bidirectional layer
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2, name="bidir_rnn", dropout=.2))(time_dense)

    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)   
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    
    return model