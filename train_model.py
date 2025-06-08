import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Lokasi dataset
image_dir = "fruit-detection-dataset/images/train"

# Ekstrak label dari nama file
def extract_label(filename):
    return filename.split('_')[0]

# Ekstrak fitur dari gambar
def extract_features(image_path, size=(32, 32)):
    with Image.open(image_path) as img:
        img = img.resize(size).convert("RGB")
        return np.array(img).flatten()

# Ambil semua gambar
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
X = []
y = []

for f in image_files:
    path = os.path.join(image_dir, f)
    X.append(extract_features(path))
    y.append(extract_label(f))

X = np.array(X)
y = np.array(y)

# Bagi data dan latih model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Simpan model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/random_forest_fruit_model.pkl")
print("Model disimpan di model/random_forest_fruit_model.pkl")

# Evaluasi model
os.makedirs("static", exist_ok=True)

# 1. Simpan laporan klasifikasi
report_text = classification_report(y_test, model.predict(X_test))
with open("static/classification_report.txt", "w") as f:
    f.write("Akurasi: {:.2f}\n\n".format(model.score(X_test, y_test)))
    f.write("Laporan Klasifikasi:\n")
    f.write(report_text)

# 2. Simpan confusion matrix ke file PNG
cm = confusion_matrix(y_test, model.predict(X_test), labels=np.unique(y))
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Prediksi")
plt.ylabel("Sebenarnya")
plt.title("Confusion Matrix - Klasifikasi Buah")
plt.tight_layout()
plt.savefig("static/confusion.png")
plt.close()
