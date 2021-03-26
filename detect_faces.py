
# import necesary library
import argparse
import cv2
import numpy as np

IMAGE = 'einbo.jpg'
PROTO = "prototxt.txt"
MODEL = 'respre.caffemodel'

print("[INFO] loading model..")
net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

img = cv2.imread(IMAGE)
(h,w) = img.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
	(300, 300))

print("[INFO] predicting...")
net.setInput(blob)
detections = net.forward()

print(detections.shape)

thresh = 0.5
# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]
	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > thresh:
		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(img, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(img, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


# show the output image
cv2.imshow("Output", img)
cv2.waitKey(0)