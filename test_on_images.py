import os
import face_recognition
import cv2
import pickle
import numpy as np 

test_dir = 'Photos/'    #directory in which testing will be done
files = os.listdir(test_dir)

label_file = open('labels.pkl', 'rb')
labels = pickle.load(label_file)
model_file = open('model.pkl', 'rb')
model = pickle.load(model_file)
label_file.close()
model_file.close()


for file in files:
    image = cv2.imread(test_dir + file)
    orig = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, boxes)

    if len(encodings)==0:
        continue
    ans = model.predict(np.asarray(encodings).reshape(-1,128))
    for i in range(len(boxes)):
        cv2.rectangle(orig, (boxes[i][3], boxes[i][0]), (boxes[i][1], boxes[i][2]), (255,0,0), 2)
        cv2.putText(orig, labels[ans[i]], (boxes[i][3], boxes[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 600,600)
    cv2.imshow('Image', orig)
    
    k = cv2.waitKey(0) &0xFF
    if k == ord('q'):   #pressing any key will brought next image while pressing 'q' will end the test
        break
                            

