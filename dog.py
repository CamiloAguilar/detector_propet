from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True,
	help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())

rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
cap = cv2.VideoCapture('videoprueba3.mp4')
scaling_factor = 1

while (cap.isOpened()):
	ret, frame = cap.read()
	frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224, 224)), 0.007843, (224, 224), 127.5)
    net.setInput(blob)
    start = time.time()
    preds = net.forward()
    end = time.time()
    print("[INFO] classification took {:.5} seconds".format(end - start))
    idxs = np.argsort(preds[0])[::-1][:5]
    print(idxs[0])
    #print(preds[0, 0, idxs[0], 2])
    print(preds[0][idxs[0]])
    a = range(0, idxs[0])
    print(a)
    # loop over the top-5 predictions and display them
    for (i, idx) in enumerate(idxs):
    	# draw the top prediction on the input image
    	if i == 0:
    		text = "Label: {}, {:.2f}%".format(classes[idx],
    			preds[0][idx] * 100)
    		cv2.putText(frame, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
    			0.7, (0, 0, 255), 2)

    	# display the predicted label + associated probability to the
    	# console
    	print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
    		classes[idx], preds[0][idx]))

	c = cv2.waitKey(1)
	if c == 27:
		break

cap.release()
out.release()
cv2.destroyAllWindows()

# dog.py --prototxt bvlc_googlenet.prototxt --model bvlc_googlenet.caffemodel --labels synset_words.txt
