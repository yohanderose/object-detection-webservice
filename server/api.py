# Webserver things
from flask import Flask, request,  json
from flask_restful import Resource, Api

# Object detection things
import numpy as np
import sys
import time
import cv2
import os
import base64

# construct the argument parse and parse the arguments
confthres = 0.3
nmsthres = 0.1


def get_labels(labels_path):
    """Extract object classes from file.

    :param labels_path: Path to class label files
    """

    # load the COCO class labels our YOLO model was trained on
    lpath = os.path.sep.join([yolo_path, labels_path])

    print(yolo_path)
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS


def get_weights(weights_path):
    """Derive the path to the YOLO model weights.

    :param weights_path: Path to weights file
    """
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath


def get_config(config_path):
    """Derive the path to the YOLO model configuration.

    :param config_path:
    """
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath


def load_model(configpath, weightspath):
    """Load our YOLO object detector trained on COCO dataset (80 classes)

    :param configpath: Path to model config
    :param weightspath: Path to model weights
    """
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net


def do_prediction(image, net, LABELS):
    """Perform object detection.

    :param image: Image to process in base64
    :param net: Object Detection model
    :param LABELS: Possible class labels
    """

    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    # print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])

                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    objects = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # Create an object describing the object -> label, confidence and bounding box
            objects.append({
                "label": LABELS[classIDs[i]],
                "accuracy": confidences[i],
                "rectangle": {
                    "left": boxes[i][0],
                    "top": boxes[i][1],
                    "width": boxes[i][2],
                    "height": boxes[i][3],
                }
            })

    return objects


def process_image(image_base64):
    """Image to perform object detection.

    :param image_base64: Image in base64
    """
    objects = []
    try:
        # Convert base64 image into numpy array, opencv readable
        nparr = np.fromstring(base64.b64decode(image_base64), np.uint8)
        # Convert from numpy image to cv2 image formats
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Convert colour scheme to RGB
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load the neural net.  Should be local to this method as its multi-threaded endpoint
        nets = load_model(CFG, Weights)
        # Fetch objects in image
        objects = do_prediction(image, nets, Labels)

    except Exception as e:
        print("Exception  {}".format(e))

    return objects


yolo_path = '.'

# Yolov3-tiny version
labelsPath = "coco.names"
cfgpath = "yolov3-tiny.cfg"
wpath = "yolov3-tiny.weights"

# Instantiate object detection params
Labels = get_labels(labelsPath)
CFG = get_config(cfgpath)
Weights = get_weights(wpath)

# Instantiate the flask api
app = Flask(__name__)
api = Api(app)


class ObjectDetector(Resource):
    """API object that describes request handling behaviour."""

    def post(self):
        """Receive an http post request with image and uuid data."""

        # Unpack image and uuid data from the request
        data = json.loads(request.json)
        # Prepare and return a response
        response = {
            "id": data["id"],
            "objects": process_image(data["image"])
            # str(type(data["image"]))
        }
        return response


# Create routes
api.add_resource(ObjectDetector, '/')

# Run the application, enabling multi-threading and disabling debug output
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5009, threaded=True,)  # debug=True)
