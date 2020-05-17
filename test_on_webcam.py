import cv2
import numpy as np
import face_recognition
import pickle

label_file = open('labels.pkl', 'rb')
labels = pickle.load(label_file)
model_file = open('model.pkl', 'rb')
model = pickle.load(model_file)
label_file.close()
model_file.close()

video_cap = cv2.VideoCapture(0)

while True:
    _, frame = video_cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = face_recognition.face_locations(image, model='hog')
    encodings = face_recognition.face_encodings(image, boxes)
    if len(encodings)>0:
        result = model.predict(np.asarray(encodings).reshape(-1,128)).tolist()
        for i in range(len(boxes)):
            cv2.rectangle(frame, (boxes[0][3], boxes[0][0]), (boxes[0][1], boxes[0][2]), (255,0,0), 2)
            cv2.putText(frame, labels[result[i]], (boxes[0][3], boxes[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    
    cv2.imshow('video', frame)
    if cv2.waitKey(1) &0xFF == ord('q'):        #press q to end the video.
        break


video_cap.release()
cv2.destroyAllWindows()
