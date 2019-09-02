import numpy as np
from preprocessing import create_datasets, create_generators
from extracting import extract_features
from model import create_model, get_conv_base
from visualizing import display_progress

base_dir: str = '/Users/Jan/developer/ML/dogs/dataset'

train, test, val = create_datasets('hello')

conv_base = get_conv_base()

train_features, train_labels = extract_features(conv_base, train, 2000)
val_features, val_labels     = extract_features(conv_base, val,   1000)
test_features, test_labels   = extract_features(conv_base, test,  1000)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
val_features   = np.reshape(val_features,   (1000, 4 * 4 * 512))
test_features  = np.reshape(test_features,  (1000, 4 * 4 * 512))

model = create_model()

model.summary()

train_generator, val_generator = create_generators(
    '/Users/Jan/Developer/ML/dogs/dataset/train',
    '/Users/Jan/Developer/ML/dogs/dataset/val',
    '/Users/Jan/Developer/ML/dogs/dataset/test'
)


history = model.fit_generator(
    
    train_generator, 
    steps_per_epoch=100, 
    epochs=30, 
    validation_data=val_generator,
    validation_steps=50
    
)

display_progress(history)

