# this code was sourced from https://github.com/openai/glow/blob/master/demo/align_face.py

# import the necessary packages
from imutils.face_utils import FaceAligner
from PIL import Image
import numpy as np
import argparse
import imutils
import dlib
import cv2
from glob import glob
from natsort import natsorted
from skimage.io import imread, imsave


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./align_faces/shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256,
                 desiredLeftEye=(0.371, 0.480))


# Input: numpy array for image with RGB channels
# Output: (numpy array, face_found)
def align_face(img):
    img = img[:, :, ::-1]  # Convert from RGB to BGR format
    img = imutils.resize(img, width=800)

    # detect faces in the grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)

    if len(rects) > 0:
        # align the face using facial landmarks
        align_img = fa.align(img, gray, rects[0])[:, :, ::-1]
        align_img = np.array(Image.fromarray(align_img).convert('RGB'))
        return align_img, True
    else:
        # No face found
        return None, False

# Input: img_path
# Output: aligned_img if face_found, else None
def align(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')  # if image is RGBA or Grayscale etc
    img = np.array(img)
    x, face_found = align_face(img)
    assert face_found == True, "no face found in the image"
    return x


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", type=str, help="path to input image", default=None)
    parser.add_argument("-output", type=str, help="path to output image", default=None)
    args   = parser.parse_args()
    img = align(args.input)
    imsave(args.output, img)
