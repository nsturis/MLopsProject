"""
 Images with issues :
     - Cat/666 : I removed it and copied 665 into a 666
     - Dog/11702 : same, I removed it and copied 11701 into a 11702
"""

import numpy as np 

import os
from os import listdir
from os.path import isfile, join

from matplotlib import image

path_data = 'C:/Users/Laurine/Downloads/PetImages/'
path_cat = path_data+'Cat/'
path_dog = path_data+'Dog/'

cat_files = [path_cat+f for f in listdir(path_cat) if isfile(join(path_cat, f))]
dog_files = [path_dog+f for f in listdir(path_dog) if isfile(join(path_dog, f))]

def npz_transformation(list_files, aNumberOfFiles, npz_filename):
    images = []
    nb_files = 0
    for f in list_files:
        images.append(np.array(image.imread(f)))
        if len(images)%100 == 0:
            print(len(images))
        # save npz of 1000 images
        if len(images)%aNumberOfFiles == 0:
            np.savez_compressed(
                f'{npz_filename}_{nb_files}', 
                images=np.array(images), 
                labels=np.array([npz_filename for _ in range (aNumberOfFiles)]))
            print(f'{npz_filename}_{nb_files} is saved')
            images = []
            nb_files+=1
    # save the last bunch of images < 1000
    np.savez_compressed(
        f'{npz_filename}_{nb_files}', 
        images=np.array(images), 
        labels=np.array([npz_filename for _ in range (len(images))]))
    print(f'{npz_filename}_{nb_files} is saved')
    print(f'DONE - {npz_filename}')
    

if __name__ == '__main__':
    nb_images_per_file = 1000
    npz_transformation(cat_files, nb_images_per_file, 'cat')
    npz_transformation(dog_files, nb_images_per_file, 'dog')