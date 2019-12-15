import numpy as np
import argparse
import cv2
import os

# Arg parse
# Example use   py .\yolo.py --image images/soccer.jpg --output output/soccer.jpg
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())


# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3),
	dtype="uint8")
print(colors[1])

# Loading image
img = cv2.imread(args["image"])
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > args["confidence"]:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

font = cv2.FONT_HERSHEY_PLAIN   
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        conf = str(round(confidences[i]*100,2))+"%"

        color = (int(colors[class_ids[i]][0]), int(colors[class_ids[i]][1]), int(colors[class_ids[i]][2]))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y), font, 0.8, (0,0,0), 3)
        cv2.putText(img, label, (x, y), font, 0.8, (255,255,255), 1)

        cv2.putText(img, conf, (x, y+h), font, 0.8, color, 1)

cv2.imwrite(args["output"], img)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()