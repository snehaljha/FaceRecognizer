import numpy as np
x = np.load('feats.npy')
y = np.load('ans_single.npy')


from sklearn.naive_bayes import GaussianNB
clas = GaussianNB()     #tweak the parameters as needed
clas.fit(x, y)

import pickle
model_file = open('model.pkl', 'wb')
pickle.dump(clas, model_file)
model_file.close()