import os
import shutil
import zipfile

# Step 1: Unzip
zip_path = 'archive (3).zip'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall()
print("✅ Zip extracted to current directory.")

# Step 2: Remove unwanted emotions
unwanted = ['fear', 'disgust', 'laugh']  # remove existing 'laugh'
base_path = os.path.join('images', 'images')

for emotion in unwanted:
    for split in ['train', 'validation']:
        folder = os.path.join(base_path, split, emotion)
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"🗑️ Deleted: {folder}")

# Step 3: Rename 'surprise' to 'laugh'
for split in ['train', 'validation']:
    surprise_folder = os.path.join(base_path, split, 'surprise')
    laugh_folder = os.path.join(base_path, split, 'laugh')
    if os.path.exists(surprise_folder):
        os.rename(surprise_folder, laugh_folder)
        print(f"🔄 Renamed: {surprise_folder} → {laugh_folder}")

# Step 4: Rename 'validation' to 'test'
old_path = os.path.join(base_path, 'validation')
new_path = os.path.join(base_path, 'test')

if os.path.exists(old_path):
    os.rename(old_path, new_path)
    print(f"📁 Renamed: {old_path} → {new_path}")
else:
    print("⚠️ 'validation' folder not found.")

# Step 5: Print structure and label map
train_dir = os.path.join(base_path, "train")
test_dir = os.path.join(base_path, "test")

train_classes = sorted(os.listdir(train_dir))
test_classes = sorted(os.listdir(test_dir))

print("📁 Train classes:", train_classes)
print("📁 Test classes:", test_classes)

label2idx = {label: idx for idx, label in enumerate(train_classes)}
print("🔢 Label to Index Mapping:", label2idx)
