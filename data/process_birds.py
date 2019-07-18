import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from skimage.io import imread, imsave
from skimage.transform import resize
import pandas  as pd
import os
from tqdm import tqdm


root_dir     = "./birds"
save_dir     = "./birds_preprocessed/"
IMG_SIZE     = 64


# utility functions -> from STACKGAN birds pre-processing code
def cropper(img, bbox):
    imsiz = img.shape
    center_x = int((2 * bbox[0] + bbox[2]) / 2)
    center_y = int((2 * bbox[1] + bbox[3]) / 2)
    R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
    y1 = np.maximum(0, center_y - R)
    y2 = np.minimum(imsiz[0], center_y + R)
    x1 = np.maximum(0, center_x - R)
    x2 = np.minimum(imsiz[1], center_x + R)
    img_cropped = img[y1:y2, x1:x2, :]
    return img_cropped

# start code here
df_images    = pd.read_csv(root_dir+"/CUB_200_2011/images.txt",header=None,
                        sep=" ",names=["id","image"])
df_train_ids = pd.read_csv(root_dir+"/CUB_200_2011/train_test_split.txt",header=None,
                        sep=" ", names=["id", "is_train"])
df_bbox      = pd.read_csv(root_dir+"/CUB_200_2011/bounding_boxes.txt",header=None,
                        sep=" ", names=["id", "x", "y", "width", "height"]
                        ).astype("int")

df = pd.merge(df_images, df_train_ids, on="id")
df = pd.merge(df, df_bbox, on="id")
df_train = df.query("is_train == 1")
df_test  = df.query("is_train == 0")

# saving training images
train_save_path = save_dir+"/train/train"
if not os.path.exists(train_save_path):
    os.makedirs(train_save_path)

train_paths = df_train.image.values
for i,path in tqdm(enumerate(train_paths), desc="processing train images"):
    img_path = root_dir + "/CUB_200_2011/images/" + path
    img      = imread(img_path)
    if len(img.shape) < 3:
        img  = np.stack([img,img,img],axis=-1)
    x,y,_    = img.shape
    bbox     = df_train.query("image == '%s'"%path)
    img_crop = cropper(img ,bbox.values[0,3:])
    img_crop = resize(img_crop, (IMG_SIZE,IMG_SIZE,3),
                      preserve_range=False, anti_aliasing=True,
                      mode="constant")
    img_crop = (img_crop*255).astype("uint8")
    imsave(train_save_path+"/%0.4d.jpg"%i, img_crop)

# saving test images
test_save_path = save_dir+"/validation/validation"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
test_paths = df_test.image.values
for i,path in tqdm(enumerate(test_paths), desc="processing test images"):
    img_path = root_dir + "/CUB_200_2011/images/" + path
    img      = imread(img_path)
    if len(img.shape) < 3:
        img  = np.stack([img,img,img],axis=-1)
    x,y,_    = img.shape
    bbox     = df_test.query("image == '%s'"%path)
    img_crop = cropper(img ,bbox.values[0,3:])
    img_crop = resize(img_crop, (IMG_SIZE,IMG_SIZE,3),
                      preserve_range=False, anti_aliasing=True,
                      mode="constant")
    img_crop = (img_crop*255).astype("uint8")
    imsave(test_save_path+"/%0.4d.jpg"%i, img_crop)
