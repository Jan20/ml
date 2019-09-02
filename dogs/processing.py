import os, shutil
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Creates train, validation and test sets
def create_datasets(base: str) -> (str, str, str):

    src: str = '/Users/Jan/Developer/ml/dogs/dataset/original_train_data'
    base: str = '/Users/Jan/Developer/ml/dogs/dataset'
    
    create_directory(base)
     
    dirs = [
        'train', 
        'val', 
        'test',
        'train/cats',
        'val/cats',
        'test/cats', 
        'train/dogs',
        'val/dogs',
        'test/dogs'
    ]

    dirs = [os.path.join(base, value) for value in dirs]
    
    [create_directory(directory) for directory in dirs]

    [copy_files([f'cat.{i}.jpg' for i in range(   0, 1000)], src, os.path.join(base, dirs[3]))]
    [copy_files([f'cat.{i}.jpg' for i in range(1000, 1500)], src, os.path.join(base, dirs[4]))]
    [copy_files([f'cat.{i}.jpg' for i in range(1500, 2000)], src, os.path.join(base, dirs[5]))]
    [copy_files([f'dog.{i}.jpg' for i in range(   0, 1000)], src, os.path.join(base, dirs[6]))]
    [copy_files([f'dog.{i}.jpg' for i in range(1000, 1500)], src, os.path.join(base, dirs[7]))]
    [copy_files([f'dog.{i}.jpg' for i in range(1500, 2000)], src, os.path.join(base, dirs[8]))]

    return dirs[0], dirs[1], dirs[2]


def create_generators(train_dir: str, val_dir: str, test_dir: str):

    # Data Generators
    train_datagen = ImageDataGenerator(
    
        rescale=1/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    
    )

    val_datagen = ImageDataGenerator(rescale=1/255)

    # Data Flow Generators
    train_generator = train_datagen.flow_from_directory(
    
        train_dir, 
        target_size=(150, 150), 
        batch_size=20, 
        class_mode='binary'
    
    )    
    
    val_generator = val_datagen.flow_from_directory(
    
        val_dir, 
        target_size=(150, 150), 
        batch_size=20, 
        class_mode='binary',
    
    )

    return train_generator, val_generator


def augment_images():
    
    cat_dir: str = '/Users/Jan/Developer/ml/dogs/dataset/train/cats'

    cat_files = [os.path.join(cat_dir, file_name) for file_name in os.listdir(cat_dir)]

    cat_file = cat_files[112]

    img = image.load_img(cat_file, target_size=(150, 150))
    
    x = image.img_to_array(img)

    x = x.reshape((1,) + x.shape)

    i = 0


    for batch in datagen.flow(x, batch_size=1):

        plt.figure(i)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i % 4 == 0:

            break
    
    plt.show()



# create_train_val_test_sets()


import os, shutil

def create_directory(dir_name: str):

    try:

        os.mkdir(dir_name)

    except IOError:
            
        print('Directory does already exist.')

def copy_files(file_names: [str], src_dir: str, dst_dir: str):

    for file_name in file_names:

        src = os.path.join(src_dir, file_name)
        dst = os.path.join(dst_dir, file_name)
        
        shutil.copyfile(src, dst)