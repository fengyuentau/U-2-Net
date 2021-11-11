import os
import argparse

from skimage import io, transform
import numpy as np
from PIL import Image
import cv2 as cv

parser = argparse.ArgumentParser(description='Demo: U2Net Inference Using OpenCV')
parser.add_argument('--input', '-i')
parser.add_argument('--model', '-m', default='u2net_human_seg.onnx')
args = parser.parse_args()

def normPred(d):
    ma = np.amax(d)
    mi = np.amin(d)
    return (d - mi)/(ma - mi)

def save_output(image_name, predict):
    img = cv.imread(image_name)
    h, w, _ = img.shape

    predict = np.squeeze(predict, axis=0)
    img_p = (predict * 255).astype(np.uint8)
    img_p = cv.resize(img_p, (w, h))
    cv.imwrite('{}-result-opencv_dnn.png'.format(image_name), img_p)

def main():
    # load net
    net = cv.dnn.readNet(args.model)

    input_size = 320 # fixed
    # build blob using OpenCV
    img = cv.imread(args.input)
    blob = cv.dnn.blobFromImage(img, scalefactor=(1.0/255.0), size=(input_size, input_size), swapRB=True)

    # build blob using others
    #img = io.imread(args.input)
    #img = transform.resize(img, (input_size, input_size), mode='constant')
    #img = np.transpose(img, (2, 0, 1))
    #blob = img[np.newaxis, :, :, :]

    # Inference
    net.setInput(blob)
    d0 = net.forward()

    # Norm
    pred = normPred(d0[:, 0, :, :])

    # Save
    save_output(args.input, pred)

if __name__ == '__main__':
    main()
