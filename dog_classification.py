# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
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
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
args = vars(ap.parse_args())

rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (224, 224)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
#vs = cv2.VideoCapture('videoprueba3.mp4')
#vs = cv2.VideoCapture('Beagle.mp4')
vs = cv2.VideoCapture('beyota1.mp4')
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	(grabbed, frame) = vs.read()
		# if the frame was not grabbed, then we have reached the
	# end of the stream
	if not grabbed:
		break

	#frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# grab the frame dimensions and convert it to a blob
	(H, W) = frame.shape[:2]
#	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224, 224)), 0.007843, (224, 224), 127.5)
	blob = cv2.dnn.blobFromImage(frame, 1, (224, 224), (104, 117, 123))

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
#	print(detections)
#	idxs = np.argsort(detections[0])[::-1][:1000]
#	test =[]
#	for i in range(0, len(idxs)):
#		test.append(detections[0][idxs[i]]*100)
#	print(test)
#	print(np.argmax(test))
#	idx = np.argmax(test)
#	print(detections[0][idx] * 100)
#	print(np.argmax(detections[0]))

	idxs = np.argsort(detections[0])[::-1][:5]
	print(max(detections[0]))

#	for i in detections[0]:

	for (i, idx) in enumerate(idxs):
		# draw the top prediction on the input image
		if i == 0:
			text = "Label: {}, {:.2f}%".format(classes[idx],
				detections[0][idx] * 100)
			cv2.putText(frame, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 0, 255), 2)

		# display the predicted label + associated probability to the
		# console
		print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
			classes[idx], detections[0][idx]))


	rects = []
#	print(idxs[0])
#	print(idxs[0]==885)
	# loop over the detections
#	for (i, idx) in enumerate(idxs):
		# extract the confidence (i.e., probability) associated with
		# the prediction
#		confidence = detections[0][idxs[0]]

#	if ((idx >= 151) & (idx <=267) & (detections[0][idx]*100>40)):
#		text = "Dog: {}, {:.2f}%".format(classes[idx],detections[0][idx] * 100)
#		cv2.putText(frame, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
#	else:
#		text = "Dog: Criollo {}, {:.2f}%".format(classes[idx], 100 - (detections[0][idx]*100))
#		cv2.putText(frame, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence


#		if confidence > args["confidence"]:
#			# extract the index of the class label from the
#			# `detections`, then compute the (x, y)-coordinates of
#			# the bounding box for the object
#			idx = int(detections[0][idxs[0]])
#			box = detections[0][idxs[0]] * np.array([W, H, W, H])
#			(startX, startY, endX, endY) = box.astype("int")
#			rects.append(box.astype("int"))
#
#			# draw the prediction on the frame
#			label = "{}".format(CLASSES[idxs[0]])
#			cv2.rectangle(frame, (startX, startY), (endX, endY),
#				COLORS[idx], 2)
#			y = startY - 15 if startY - 15 > 15 else startY + 15
#			cv2.putText(frame, label, (startX, y),
#				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idxs[0]], 2)

	objects = ct.update(rects)
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "{}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


# dog_classification.py --prototxt bvlc_googlenet.prototxt --model bvlc_googlenet.caffemodel --labels synset_words.txt --output webcam_face_recognition_output.avi
