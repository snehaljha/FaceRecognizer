import dlib
import face_recognition
import cv2
import os
import numpy as np
import pickle

labels = {}
training_dir = 'Training/'   #directory containing images of different people in different folder
names = os.listdir(training_dir)  
x = []
y = []
y_ind = []


for i in range(len(names)):
    labels[i] = names[i]
    files = os.listdir(training_dir + names[i])
    for file in files:
        image = cv2.imread(training_dir + names[i] + '/' + file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, boxes)

        for encoding in encodings:
            tmp = [0]*len(names)
            tmp[i] = 1
            y.append(tmp)
            y_ind.append(i)
            x.append([encoding])

x = np.asarray(x)
y = np.asarray(y)
y_ind = np.asarray(y_ind)

np.save('feats.npy', x)
np.save('ans_single.npy', y_ind)
np.save('ans_multi.npy', y)


label_file = open('labels.pkl', 'wb')
pickle.dump(labels, label_file)
label_file.close()