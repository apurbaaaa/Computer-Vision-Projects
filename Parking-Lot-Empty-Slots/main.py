import os
from skimage.io import imread #scikit image used
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
input_directory = './clf-data'
categories = ['empty', 'not_empty']

data = []
labels = []

for category_idx, category in enumerate(categories): #enumerate function returns index for each label
    for file in os.listdir(os.path.join(input_directory, category)):
        img_path = os.path.join(input_directory, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten()) #to one-d image
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle = True, stratify = labels )

model = SVC()

parameters = [{'gamma' : [0.01, 0.001, 0.0001], 'C' : [10, 100, 1000]}]
grid_search = GridSearchCV(model, parameters)

grid_search.fit(X_train, y_train)

best_estimator = grid_search.best_estimator_

prediction = best_estimator.predict(X_test)

print("Accuracy = ", accuracy_score(prediction, y_test))

pickle.dump(best_estimator, open('./model.p', 'wb'))