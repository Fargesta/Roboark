import cv2 as cv
import numpy as np

class DetectionEngine:
    colors = None
    ln = None
    net = None
    labels = None
    W = None
    H = None

    def __init__(self, net, labels=None):
        self.net = net
        self.labels = labels

        np.random.seed(123)
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_with_boxes(self, frame, confidence_threshold=0.5, overlapping_threshold=0.3):
        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]

        # Construct blob of frames by standardization, resizing, and swapping Red and Blue channels (RBG to RGB)
        blob = cv.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)
        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confidence_threshold:
                    # Scale the bboxes back to the original image size
                    box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # Remove overlapping bounding boxes and boundig boxes
        bboxes = cv.dnn.NMSBoxes(
            boxes, confidences, confidence_threshold, overlapping_threshold)
        if len(bboxes) > 0:
            for i in bboxes.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in self.colors[classIDs[i]]]
                frame = np.array(frame[:,::-1])
                cv.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                text = "{}: {:.4f}".format(self.labels[classIDs[i]], confidences[i])
                cv.putText(frame, text, (x, y - 5),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    def detect_with_confidence(self, frame, confidence_threshold=0.5):
        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]

        # Construct blob of frames by standardization, resizing, and swapping Red and Blue channels (RBG to RGB)
        blob = cv.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)
        # confidences = []
        # classIDs = []
        resultList = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confidence_threshold:
                    resultList.append(tuple((classID, self.labels[classID], float(confidence))))
        
        return resultList
