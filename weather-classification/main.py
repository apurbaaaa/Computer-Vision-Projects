import os
from img2vec_pytorch import Img2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
#import data and prepare
img2vec = Img2Vec()

data_dir = './data/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

data = {}
for j, dir_ in enumerate([train_dir, test_dir]):
    features = []
    labels = []
    for category in os.listdir(dir_):
        for img_path in os.listdir(os.path.join(dir_, category)):
            img_path_ = os.path.join(dir_, category, img_path)
            img = Image.open(img_path_)

            img_features = img2vec(img)

            features.append(img_features)
            labels.append(category)

    data[['training_data', 'validation_data'][j]] = features
    data[['training_labels', 'validation_labels'][j]] = labels
#train model
model = RandomForestClassifier()
model.fit(data['training_data'], data['training_labels'])
#test model

#save model