from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.applications import VGG16

def create_model() -> Sequential: 
    
    conv_base = get_conv_base()
    conv_base.trainable = False

    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc'])

    model.summary()

    return model

def get_conv_base() -> VGG16:

    return VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

