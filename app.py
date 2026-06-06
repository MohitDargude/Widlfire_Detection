# # app.py
# from flask import Flask, render_template, request, redirect, url_for
# import os
# from utils.predict import predict_wildfire

# import os
# print(os.path.abspath("app/templates"))


# app = Flask(__name__, template_folder="app/templates")
# app.config['UPLOAD_FOLDER'] = 'uploads/'

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
#     if file:
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filepath)

#         print(f"File saved at: {filepath}")  # Check file path
#         print(f"Uploaded file exists? {os.path.exists(filepath)}")  # Check if file was saved

#         prediction = predict_wildfire(filepath)
#         return render_template('index.html', prediction=prediction, image_url=filepath)

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, render_template

# app = Flask(__name__, template_folder="app/templates")  # Explicitly specify the template folder path

# @app.route('/')
# def index():
#     return render_template('index.html')

# if __name__ == "__main__":
#     app.run(debug=True)


# app.py
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2
import base64
import os

app = Flask(__name__, template_folder='app/templates')

# Load the trained model
try:
    model = tf.keras.models.load_model('models/wildfire_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(image):
    try:
        # Resize image to match model input size
        image = image.resize((224, 224))
        # Convert to array and preprocess
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({
            'prediction': 'Error: No image provided',
            'confidence': 0
        }), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'prediction': 'Error: No selected file',
            'confidence': 0
        }), 400
    
    try:
        # Read and preprocess the image
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return jsonify({
                'prediction': 'Error: Failed to process image',
                'confidence': 0
            }), 500

        if model is None:
            return jsonify({
                'prediction': 'Error: Model not loaded',
                'confidence': 0
            }), 500
        
        # Make prediction
        prediction = model.predict(processed_image)[0][0]
        confidence = float(prediction)
        
        # Determine result
        result = "Wildfire Detected" if confidence > 0.5 else "No Wildfire"
        
        return jsonify({
            'prediction': result,
            'confidence': confidence
        })
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({
            'prediction': f'Error: {str(e)}',
            'confidence': 0
        }), 500

@app.route('/detect_live', methods=['POST'])
def detect_live():
    if 'file' not in request.files:
        return jsonify({
            'prediction': 'Error: No image provided',
            'confidence': 0
        }), 400

    try:
        file = request.files['file']
        # Convert to PIL Image
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return jsonify({
                'prediction': 'Error: Failed to process image',
                'confidence': 0
            }), 500

        if model is None:
            return jsonify({
                'prediction': 'Error: Model not loaded',
                'confidence': 0
            }), 500
        
        # Make prediction
        prediction = model.predict(processed_image)[0][0]
        confidence = float(prediction)
        
        # Determine result
        result = "Wildfire Detected" if confidence > 0.5 else "No Wildfire"
        
        return jsonify({
            'prediction': result,
            'confidence': confidence
        })
    except Exception as e:
        print(f"Error in live detection: {e}")
        return jsonify({
            'prediction': f'Error: {str(e)}',
            'confidence': 0
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
