from flask import Flask, render_template, request
import os
from PIL import Image
import numpy as np
import joblib

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = joblib.load('model/random_forest_fruit_model.pkl')

# Ekstrak fitur dari gambar
def extract_features(image_path, size=(32, 32)):
    with Image.open(image_path) as img:
        img = img.resize(size).convert("RGB")
        return np.array(img).flatten().reshape(1, -1)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            features = extract_features(filepath)
            prediction = model.predict(features)[0]

    return render_template('index.html', prediction=prediction, filename=filename)

@app.route('/report')
def report():
    report_path = os.path.join("static", "classification_report.txt")
    with open(report_path, "r") as f:
        report_text = f.read()
    return render_template("report.html", report=report_text)

if __name__ == '__main__':
    app.run(debug=True)
