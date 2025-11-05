import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore")

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
X, y = [], []

for f in image_files:
    path = os.path.join(image_dir, f)
    X.append(extract_features(path))
    y.append(extract_label(f))

X = np.array(X)
y = np.array(y)

# Bagi data dan latih model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Simpan model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/random_forest_fruit_model.pkl")
print("✅ Model disimpan di model/random_forest_fruit_model.pkl")

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
sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt="d",
    xticklabels=np.unique(y),
    yticklabels=np.unique(y),
)
plt.xlabel("Prediksi")
plt.ylabel("Sebenarnya")
plt.title("Confusion Matrix - Klasifikasi Buah")
plt.tight_layout()
plt.savefig("static/confusion.png")
plt.close()

# 3. ROC Curve dan AUC (versi aman)
print("📈 Menghitung ROC Curve dan AUC...")
try:
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    y_score = model.predict_proba(X_test)
    n_classes = y_test_bin.shape[1]

    fpr, tpr, roc_auc = {}, {}, {}
    valid_classes = []

    for i in range(n_classes):
        if np.sum(y_test_bin[:, i]) > 1:  # pastikan ada sample positif
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            valid_classes.append(i)

    if len(valid_classes) == 0:
        print("⚠️ Tidak ada kelas yang cukup untuk dihitung ROC.")
    else:
        plt.figure(figsize=(7, 6))
        for i in valid_classes:
            plt.plot(fpr[i], tpr[i], label=f'{np.unique(y)[i]} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Klasifikasi Buah')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig("static/roc_curve.png")
        plt.close()

        mean_auc = np.mean(list(roc_auc.values()))
        with open("static/classification_report.txt", "a") as f:
            f.write("\nRata-rata AUC: {:.3f}\n".format(mean_auc))
        print("✅ ROC curve disimpan di static/roc_curve.png")
        print("✅ Rata-rata AUC:", round(mean_auc, 3))

except Exception as e:
    print("❌ Gagal menghitung ROC/AUC:", e)
