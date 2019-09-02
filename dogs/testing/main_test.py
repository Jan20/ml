import os

import matplotlib.pyplot as plt

#
# This cannot be the quickest way to get the parent directory
#

current_dir = os.getcwd()
current_dir = current_dir.split('/')

parent_dir: str = ''
parent_dir_list: [str] = []

[parent_dir_list.append(element) for element in current_dir if element != 'testing' and element != '']

for element in parent_dir_list:

    parent_dir = parent_dir.__add__(f'/{element}')

parent_dir = parent_dir + '/dataset'

# Quick overview over the data that is used for training, validation and testing.

print(f'total training cat images: {len(os.listdir(parent_dir + "/train/cats"))}')
print(f'total training dog images: {len(os.listdir(parent_dir + "/train/dogs"))}')
print(f'total training cat images: {len(os.listdir(parent_dir + "/val/cats"))}')
print(f'total training dog images: {len(os.listdir(parent_dir + "/val/dogs"))}')
print(f'total training cat images: {len(os.listdir(parent_dir + "/test/cats"))}')
print(f'total training dog images: {len(os.listdir(parent_dir + "/test/dogs"))}')


    # for data_batch, labels_batch in train_generator:

    #     print(f'Data batch shape: {data_batch.shape}')
    #     print(f'Labels batch shape: {labels_batch.shape}')
