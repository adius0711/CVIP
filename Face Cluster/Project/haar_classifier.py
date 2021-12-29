import os
import cv2
import sys
import json
import imutils
import numpy as np

detector = cv2.CascadeClassifier("./Model_Files/haarcascade_frontalface_default.xml")

path = ""
json_list = []

cmdLine_Args = sys.argv
print('cmdLine_Args', cmdLine_Args, )
for i in range(1, len(cmdLine_Args)):
    if i == 1:
        path = path + cmdLine_Args[i]
    else:
        path = path + " " + cmdLine_Args[i]

imagePath = input('PLease enter the image folder path ')
path = os.path.join(path, imagePath)

for imageName in os.listdir(path):
    imgPath = (path + "/" + imageName)
    image = cv2.imread(imgPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray, scaleFactor=1.06,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        # draw the face bounding box on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        element = {"iname": imageName, "bbox": [int(x), int(y), int(w), int(h)]}
        json_list.append(element)

    cv2.imshow("Image", image)
    cv2.waitKey(0)

output_json = "results_a.json"

# dump json_list to result.json
with open(output_json, 'w') as f:
    json.dump(json_list, f)
