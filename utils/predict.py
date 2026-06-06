# # utils/predict.py
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from .preprocess import preprocess_image

# # Load the trained model
# MODEL_PATH = './models/wildfire_model.h5'
# model = load_model(MODEL_PATH)

# def predict_wildfire(img_path):
#     """
#     Predicts whether an image contains a wildfire.

#     Args:
#         img_path (str): Path to the image file.

#     Returns:
#         str: Prediction result ("Wildfire Detected" or "No Wildfire Detected").
#     """
#     # Preprocess the image
#     img_array = preprocess_image(img_path)
    
#     # Make prediction using the model
#     prediction = model.predict(img_array)
    
#     # Interpret the prediction
#     if prediction[0][0] > 0.5:
#         return "Wildfire Detected"
#     else:
#         return "No Wildfire Detected"


#ChatGPT code starts

# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from .preprocess import preprocess_image

# # Load the trained model with error handling
# MODEL_PATH = './models/wildfire_model.h5'
# try:
#     model = load_model(MODEL_PATH)
#     print("✅ Model loaded successfully!")
# except Exception as e:
#     print(f"❌ Error loading model: {e}")

# def predict_wildfire(img_path, threshold=0.4):
#     """
#     Predicts whether an image contains a wildfire.

#     Args:
#         img_path (str): Path to the image file.
#         threshold (float): Decision threshold for wildfire detection.

#     Returns:
#         str: Prediction result ("Wildfire Detected" or "No Wildfire Detected").
#     """
#     # Preprocess the image
#     img_array = preprocess_image(img_path)
    
#     # Make prediction using the model
#     prediction = model.predict(img_array)
#     prediction_score = prediction[0][0]

#     # Debugging: Print prediction score
#     print(f"🔍 Prediction Score: {prediction_score:.4f}")

#     # Interpret the prediction
#     if prediction_score > threshold:
#         return "🔥 Wildfire Detected"
#     else:
#         return "✅ No Wildfire Detected"


import tensorflow as tf
from tensorflow.keras.models import load_model
from .preprocess import preprocess_image

MODEL_PATH = './models/wildfire_model.h5'

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

def predict_wildfire(img_path, threshold=0.4):
    """
    Predicts whether an image contains a wildfire.

    Args:
        img_path (str): Path to the image file.
        threshold (float): Decision threshold for wildfire detection.

    Returns:
        str: Prediction result ("Wildfire Detected" or "No Wildfire Detected").
    """
    img_array = preprocess_image(img_path)
    
    # Make prediction
    prediction = model.predict(img_array)
    prediction_score = prediction[0][0]  # Assuming a single output neuron
    
    print(f"🔍 **Raw Prediction Score:** {prediction_score}")  # Debugging step

    if prediction_score > threshold:
        return "🔥 Wildfire Detected"
    else:
        return "✅ No Wildfire Detected"
