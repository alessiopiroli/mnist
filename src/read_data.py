import os
import torch
import numpy as np

# training data
training_data_path = os.path.join('data', 'train-images-idx3-ubyte')
training_labels_path = os.path.join('data', 'train-labels-idx1-ubyte')

# testing data
test_data_path = os.path.join('data','t10k-images-idx3-ubyte')
test_labels_path = os.path.join('data', 't10k-labels-idx1-ubyte')

def get_int_val(raw_bytes):
    return int.from_bytes(raw_bytes, 'big')

def get_training_data():
    training_images = torch.empty((0,0))
    training_labels = torch.empty((0,0))
    try:
        # getting training images
        with open(training_data_path, 'rb') as file:
            if get_int_val(file.read(4)) == 2051:
                num_img = get_int_val(file.read(4))
                img_heigth = get_int_val(file.read(4))
                img_width = get_int_val(file.read(4))
                image_file_buffer = bytearray(file.read())
                training_images = torch.frombuffer(image_file_buffer, dtype=torch.uint8)
                training_images = training_images.view(num_img, img_heigth, img_width)

            else:
                print("Something went wrong in the training data reading")

        # getting training labels
        with open(training_labels_path, 'rb') as file:
            if get_int_val(file.read(4)) == 2049:
                num_labels = get_int_val(file.read(4))
                buffer = bytearray(file.read())
                training_labels = torch.frombuffer(buffer, dtype=torch.uint8)

            else:
                print("Something went wrong in the training labels reading")

        return training_images, training_labels

    except FileNotFoundError:
        print("Error in opening the files")

def get_test_data():
    test_images = torch.empty((0,0))
    test_labels = torch.empty((0,0))
    try:
        # getting training images
        with open(test_data_path, 'rb') as file:
            if get_int_val(file.read(4)) == 2051:
                num_img = get_int_val(file.read(4))
                img_heigth = get_int_val(file.read(4))
                img_width = get_int_val(file.read(4))
                image_file_buffer = bytearray(file.read())
                test_images = torch.frombuffer(image_file_buffer, dtype=torch.uint8)
                test_images = test_images.view(num_img, img_heigth, img_width)

            else:
                print("Something went wrong in the training data reading")

        # getting training labels
        with open(test_labels_path, 'rb') as file:
            if get_int_val(file.read(4)) == 2049:
                num_labels = get_int_val(file.read(4))
                buffer = bytearray(file.read())
                test_labels = torch.frombuffer(buffer, dtype=torch.uint8)

            else:
                print("Something went wrong in the training labels reading")

        return test_images, test_labels

    except FileNotFoundError:
        print("Error in opening the files")