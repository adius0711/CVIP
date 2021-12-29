import cv2
import argparse
import os
import numpy as np
import json
from colab import files

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


def compute_images(image, confidence, detections):
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (sX, sY, eX, eY) = box.astype("int")

    cv2.rectangle(image, (sX, sY), (eX, eY),
                  (0, 0, 255), 2)
    element = {"iname": imagePath[9:], "bbox": [int(sX), int(sY), int(eX - sX), int(eY - sY)]}
    json_list.append(element)


ap = argparse.ArgumentParser()
ap.add_argument("-C", "--confidence", type=float, default=0.58,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

path = input("Please enter the path ")
imagePaths = list(list_files(path, validExts=image_types, contains=None))
d_n_net = cv2.dnn.readNetFromCaffe("./deploy.prototxt.txt", "./res10_300x300_ssd_iter_140000.caffemodel")
json_list = []

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    d_n_net.setInput(blob)
    detections = d_n_net.forward()

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            compute_images(image, confidence, detections)
    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)

# the result json file name
print(len(json_list))
output_json = "results3.json"
# dump json_list to result.json
with open(output_json, 'w') as f:
    json.dump(json_list, f)
