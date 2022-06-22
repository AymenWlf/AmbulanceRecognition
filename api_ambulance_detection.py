from flask import Flask, request
import json
import cv2 as cv
import numpy as np

app = Flask(__name__)

# Initialize the parameters
state = False;
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4  #Non-maximum suppression threshold

inpWidth = 416  #608     #Width of network's input image
inpHeight = 416 #608     #Height of network's input image

# Load names of classes
classesFile = "classes.names";

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.

modelConfiguration = "darknet/output/darknet-yolov3.cfg";
modelWeights = "weights/darknet-yolov3_last.weights";

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        print("out.shape : ", out.shape)
        for detection in out:
            #if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            #if scores[classId]>confThreshold:
            confidence = scores[classId]
            if detection[4]>confThreshold:
                print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
                print(detection)
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                state = True

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    return state



@app.route('/predict',methods=['POST'])
def predict():
    image = request.files['file']
    #cap = cv.VideoCapture(image)
    image_byte = image.read()
    decoded = cv.imdecode(np.frombuffer(image_byte, np.uint8), -1)
    #
    # # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(decoded, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    #
    # # Sets the input to the network
    net.setInput(blob)
    #
    # # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))
    #
    # # Remove the bounding boxes with low confidence
    s = postprocess(decoded, outs)
    return str(s)

if __name__ == "__main__":
    app.run(debug=True)