from tkinter import *
from tkinter import messagebox
import tkinter.filedialog as fdialog
import cv2
import os
import face_recognition
import pickle
import numpy as np

labels = None
model = None
def help():
    messagebox.showinfo('Help', 'Load Model will load previously trained model if present.\nTrain Model requires you to specify folder containing multiple folders works as label.\nTesting can done on a single image or folder containing multiple images\n\nBuild by Snehal Jha')


def loadModel():
    try:
        global labels
        global model
        labels_file = open('labels.pkl', 'rb')
        model_file = open('model.pkl', 'rb')
        labels = pickle.load(labels_file)
        model = pickle.load(model_file)
        labels_file.close()
        model_file.close()
        messagebox.showinfo('Model Status', 'Model loaded successfully')
    except:
        messagebox.showinfo('Model Status', 'Loading of model failed')


def trainModel():
    try :
        dir = fdialog.askdirectory()
        names = os.listdir(dir)
        global labels
        labels = {}
        x = []
        y = []
        for i in range(len(names)):
            labels[i] = names[i]
            files = os.listdir(dir + '/' + names[i])
            for file in files:
                image = cv2.imread(dir + '/' + names[i] + '/' + file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(image)
                encodings = face_recognition.face_encodings(image, boxes)
                for encoding in encodings:
                    x.append(encoding)
                    y.append(i)
        
        x = np.asarray(x)
        y = np.asarray(y)

        global model
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        model.fit(x,y)

        labels_file = open('labels.pkl', 'wb')
        model_file = open('model.pkl', 'wb')
        pickle.dump(labels, labels_file)
        pickle.dump(model, model_file)
        messagebox.showinfo('Model Status', 'Model trained successfully')
    except:
        messagebox.showerror('Model Status', 'Model training failes')


def pehchan(file):
    image = cv2.imread(file)
    orig = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, boxes)
    if len(encodings)>0:
        ans = model.predict(np.asarray(encodings).reshape(-1, 128))
        for i in range(len(boxes)):
            cv2.rectangle(orig, (boxes[i][3], boxes[i][0]), (boxes[i][1], boxes[i][2]), (255,0,0), 2)
            cv2.putText(orig, labels[ans[i]], (boxes[i][3], boxes[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 600, 600)
    cv2.imshow('Image', orig)

def imageRec() :
    file = fdialog.askopenfilename()
    pehchan(file)


def mImageRec() :
    dir = fdialog.askdirectory()
    files = os.listdir(dir)
    for file in files:
        pehchan(dir + '/' + file)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()
    messagebox.showinfo('Info', 'Photo queue is ended')

root = Tk()

frame1 = Frame(root, borderwidth=2)
frame1.pack(side = TOP, fill=X, expand=True)
helpbtn = Button(frame1, text="?", command=help)
helpbtn.pack(side=RIGHT)


frame2 = Frame(root, borderwidth=2)
frame2.pack(side=TOP, fill=X, expand=True)
loadbtn = Button(frame2, text='Load Model', command=loadModel)
loadbtn.pack(side=LEFT, fill=X, pady=5, padx=5, expand=True)
trainbtn = Button(frame2, text='Train Model', command=trainModel)
trainbtn.pack(side=LEFT, fill=X, expand=True, pady=5, padx=5)


frame3 = Frame(root, borderwidth=2)
frame3.pack(side=TOP, fill=X, expand=True)
imgtestbtn = Button(frame3, text='Test on Image', command=imageRec)
imgtestbtn.pack(side=LEFT, fill=X, expand=True, padx=5, pady=5)
dirtestbtn = Button(frame3, text='Test on Folder', command=mImageRec)
dirtestbtn.pack(side=LEFT, fill=X, expand=True, padx=5, pady=5)

root.mainloop()