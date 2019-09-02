from model import create_model
from training import train_model
from processing import create_generators
from visualizer import display_progress

def execute():

    model = create_model()

    base_dir: str = '/Users/Jan/Developer/ML/dogs/dataset'

    train_generator, val_generator = create_generators(
        
        f'{base_dir}/train', 
        f'{base_dir}/val', 
        f'{base_dir}/test'
    
    )

    history = train_model(train_generator, val_generator, model)

    display_progress(history)

    print(history)


execute()