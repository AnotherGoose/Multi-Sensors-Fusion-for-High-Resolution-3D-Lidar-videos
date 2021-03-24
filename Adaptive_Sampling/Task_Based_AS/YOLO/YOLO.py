import cv2
import os
import numpy as np

baseName = os.path.basename(__file__)
dir = __file__
dir = dir[:-(len(baseName))]

os.chdir(dir)
cDir = os.path.abspath(os.getcwd())

def getYOLOPredsImg(img, conThresh = 0.5, overThresh = 0.3, cuda = False):

    yoloConfigPath = cDir + '/Model_Data/yolov4.cfg'
    yoloWeightsPath = cDir + '/Model_Data/yolov4.weights'

    net = cv2.dnn.readNetFromDarknet(yoloConfigPath, yoloWeightsPath)

    if cuda:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    cocoLabelsPath = cDir + '/coco/coco.names'

    #Read Labels
    with open(cocoLabelsPath, 'r', encoding='utf-8') as f:
        labels = f.read().strip().split('\n')

    # Get layer names that output predictions from YOLO
    # List of colors to represent each class label with distinct color
    np.random.seed(1)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    W = None
    H = None


    if img is None:
        print("Error opening file")
        return

    # frame = imutils.resize(frame, width=640)

    # frame = imutils.resize(frame, width=640)
    if W is None or H is None:
        (H, W) = img.shape[:2]

    # Construct blob of frames by standardization, resizing, and swapping Red and Blue channels (RBG to RGB)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conThresh:
                # Scale the bboxes back to the original image size
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                if x < 0:
                    w = width + x
                    x = 0
                if y < 0:
                    h = height + y
                    y = 0

                if (x + width) > W:
                    width = W - x

                if (y + height) > H:
                    height = H - y

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Remove overlapping bounding boxes and bounding boxes
    bboxes = cv2.dnn.NMSBoxes(boxes, confidences, conThresh, overThresh)

    ROI = 0

    if len(bboxes) > 0:
        #Make a clean 2D array depending on ROI's provided
        ROI = np.empty(((len(bboxes.flatten())), 4), int)
        c = 0
        for i in bboxes.flatten():
            ROI[c] = boxes[i]
            c += 1
            #get bbox array

    display = True

    preds = 0
    if display == True:
        preds = img
        if len(bboxes) > 0:
            for i in bboxes.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(preds, (x, y), (x + w, y + h), color, 1)
                text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                cv2.putText(preds, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    #Check if anything was detected
    try:
        x, y, w, h = ROI[0]
        detected = True
    except:
        detected = False

    return detected, ROI, preds