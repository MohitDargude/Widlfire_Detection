# app/routes.py
from flask import Blueprint, render_template, request, redirect, url_for
import os
from utils.predict import predict_wildfire

# Create a Blueprint for the main routes
main_routes = Blueprint('main_routes', __name__)

@main_routes.route('/')
def index():
    """
    Renders the homepage where users can upload an image.
    """
    return render_template('index.html')

@main_routes.route('/upload', methods=['POST'])
def upload_image():
    """
    Handles image upload and prediction.
    """
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the uploaded file to the uploads folder
        filepath = os.path.join(main_routes.root_path, '..', 'uploads', file.filename)
        file.save(filepath)
        
        # Get the prediction from the model
        prediction = predict_wildfire(filepath)
        
        # Render the result on the homepage
        return render_template('index.html', prediction=prediction, image_url=file.filename)