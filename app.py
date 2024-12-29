from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import base64
import matplotlib

# Use the 'Agg' backend for Matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model = tf.keras.models.load_model("models\DenseNet169.h5")
class_labels = ["Mild", "Moderate", "No_dr", "Proliferate_dr", "Severe"]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    image = cv2.resize(image, (224, 224))  # Resize to match the model's input size
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Expand dimensions for batch size
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess and predict
            image = preprocess_image(filepath)
            predictions = model.predict(image)
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_labels[predicted_class_index]
            confidence = predictions[0][predicted_class_index]

            # Display the image with the prediction
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title(f"Predicted class: {predicted_class}\nConfidence: {confidence:.2f}")
            plt.axis("off")

            # Convert plot to PNG image and then to base64 for HTML rendering
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            return render_template('result.html', predicted_class=predicted_class, confidence=confidence, img_str=img_str)

    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)