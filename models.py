import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

def create_mlp(dim, regress=False):
    # simple multi-layer perceptron
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))
    
    # if it's a regression problem, add a linear node
    if regress:
        model.add(Dense(1, activation="linear"))
        
    return model

def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    # initialize the model along with the input shape
    inputShape = (height, width, depth)
    chanDim = -1
    
    inputs = tf.keras.Input(shape=inputShape)
    
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        if i == 0:
            x = inputs
            
        x = Conv2D(f, (3, 3), padding="same", activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.5)(x)
    
    x = Dense(4, activation="relu")(x)
    
    # if regress, add linear node
    if regress:
        x = Dense(1, activation="linear")(x)
        
    model = tf.keras.Model(inputs, x)
    return model
