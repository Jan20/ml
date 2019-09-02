import keras
from model import create_model
from preprocessing import create_generators

def train_model(train_generator, val_generator, model: keras.engine.sequential.Sequential) -> dict:

    history = model.fit_generator(
        
        train_generator, 
        steps_per_epoch=100, 
        epochs=5, 
        validation_data=val_generator, 
        validation_steps=50
    
    )

    model.save('cats_and_dogs.h5')

    return history