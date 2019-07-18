import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.io import loadmat
import skimage.io as sio
import os
from tqdm import tqdm

# Constants
root_dir     = "./flowers"

# loading train and test files
train_files = np.load(root_dir+"/flower_train_files.npy")
test_files  = np.load(root_dir+"/flower_test_files.npy")

images_path = glob(root_dir+"/images/*.jpg")
assert len(images_path) == len(train_files) + len(test_files), "issue in total image count"

train_files   = [root_dir+"/images/"+p for p in train_files]
test_files    = [root_dir+"/images/"+p for p in test_files]

# saving training images
train_dir = "./flowers_processed/train/train"
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
for path in tqdm(train_files, desc ="saving train images"):
    image = sio.imread(path)
    name  = path.split("/")[-1]
    sio.imsave(train_dir + "/" + name, image)

# saving test images
test_dir = "./flowers_processed/validation/validation"
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
for path in tqdm(test_files, desc="saving test images"):
    image = sio.imread(path)
    name  = path.split("/")[-1]
    sio.imsave(test_dir + "/" + name, image)
