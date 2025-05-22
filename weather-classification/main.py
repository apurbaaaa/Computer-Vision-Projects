import os
from img2vec_pytorch import Img2Vec
from sklearn.metrics import accuracy_score
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import pickle

# Initialize image vectorizer
img2vec = Img2Vec()

# Set data paths
data_dir = './data/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Dictionary to hold processed data
data = {}

# Loop through both training and test directories
for j, dir_ in enumerate([train_dir, test_dir]):
    features = []
    labels = []

    for category in os.listdir(dir_):
        category_path = os.path.join(dir_, category)
        if not os.path.isdir(category_path):
            continue

        for img_file in os.listdir(category_path):
            img_path = os.path.join(category_path, img_file)

            try:
                img = Image.open(img_path).convert('RGB')
                img_features = img2vec.get_vec(img)
                features.append(img_features)
                labels.append(category)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")


    if j == 0:
        data['training_data'] = features
        data['training_labels'] = labels
    else:
        data['validation_data'] = features
        data['validation_labels'] = labels

# Train model
model = RandomForestClassifier()
model.fit(data['training_data'], data['training_labels'])

# Test model
y_pred = model.predict(data['validation_data'])  # FIXED: pass actual feature list
print('Accuracy =', accuracy_score(data['validation_labels'], y_pred))

with open('./model.p', 'wb') as file:
    pickle.dump(model, file)
    file.close()
