import tensorflow as tf
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave
from tqdm import tqdm
import os

files = glob("./celeba/celeba-tfr/train/*.tfrecords")
save_train_dir = "./celeba_preprocessed/train/train"
os.makedirs(save_train_dir)
i = 0
train_attributes = []
train_labels     = []
for file in tqdm(files, desc="processing training files"):
    record_iterator = tf.python_io.tf_record_iterator(path=file)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        attributes = example.features.feature['attr'].int64_list.value
        label      = example.features.feature['label'].int64_list.value[0]
        shape      = example.features.feature['shape'].int64_list.value
        raw_img    = example.features.feature['data'].bytes_list.value[0]
        img        = np.frombuffer(raw_img, dtype=np.uint8)
        img        = img.reshape(shape)
        
        imsave(save_train_dir+"/%0.7d.jpg"%i,img)
        train_attributes.append(attributes)
        train_labels.append(label)
        i = i+1


files = glob("./celeba/celeba-tfr/validation/*.tfrecords")
save_test_dir = "./celeba_preprocessed/validation/validation"
os.makedirs(save_test_dir)
i = 0
test_attributes = []
test_labels     = []
for file in tqdm(files, desc="processing testing files"):
    record_iterator = tf.python_io.tf_record_iterator(path=file)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        attributes = example.features.feature['attr'].int64_list.value
        label      = example.features.feature['label'].int64_list.value[0]
        shape      = example.features.feature['shape'].int64_list.value
        raw_img    = example.features.feature['data'].bytes_list.value[0]
        img        = np.frombuffer(raw_img, dtype=np.uint8)
        img        = img.reshape(shape)
        
        imsave(save_test_dir+"/%0.7d.jpg"%i,img)
        test_attributes.append(attributes)
        test_labels.append(label)
        i = i+1
        
train_attributes = np.array(train_attributes)
test_attributes = np.array(test_attributes)
train_labels    = np.array(train_labels)
test_labels    = np.array(test_labels)
np.save("./celeba_preprocessed/train_attributes.npy", train_attributes)
np.save("./celeba_preprocessed/test_attributes.npy", test_attributes)
np.save("./celeba_preprocessed/train_labels.npy", train_labels)
np.save("./celeba_preprocessed/test_labels.npy", test_labels)
