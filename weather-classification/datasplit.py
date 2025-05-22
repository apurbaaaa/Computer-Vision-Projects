import os
import shutil
from sklearn.model_selection import train_test_split

source_dir = 'data'
train_dir = 'train'
test_dir = 'test'
test_ratio = 0.2  # 20% for test

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)
    if not os.path.isdir(category_path):
        continue

    files = os.listdir(category_path)
    train_files, test_files = train_test_split(files, test_size=test_ratio, random_state=42)

    # Create destination folders
    train_category_path = os.path.join(train_dir, category)
    test_category_path = os.path.join(test_dir, category)
    os.makedirs(train_category_path, exist_ok=True)
    os.makedirs(test_category_path, exist_ok=True)

    # Move training files
    for file in train_files:
        src = os.path.join(category_path, file)
        dst = os.path.join(train_category_path, file)
        shutil.copy2(src, dst)

    # Move testing files
    for file in test_files:
        src = os.path.join(category_path, file)
        dst = os.path.join(test_category_path, file)
        shutil.copy2(src, dst)

print("Data split into train/ and test/ successfully.")
