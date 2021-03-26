
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keyboard


net = cv2.dnn.readNetFromCaffe('prototxt.txt', 'respre.caffemodel')

cap = cv2.VideoCapture(0)

thresh = 0.5
while(True):
    
    ret, img = cap.read()
    (h,w) = img.shape[:2]

    img_b = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)), 1, (300,300))
    net.setInput(img_b)
    detections = net.forward()

    for i in range(detections.shape[2]):

    	prob = detections[0,0,i,2]

    	if prob > thresh:

    		box = detections[0,0,i,3:7] * np.array([w,h,w,h])
    		(startX, startY, endX, endY) = box.astype('int')
    		text = "{:.2f}%".format(prob * 100)
    		y = startY - 10 if startY - 10 > 10 else startY + 10
    		cv2.rectangle(img, (startX, startY), (endX, endY), (0,0,255),2)
    		cv2.putText(img, text, (startX,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)

    

    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

	