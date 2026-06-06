# # utils/preprocess.py
# import numpy as np
# from tensorflow.keras.preprocessing import image

# def preprocess_image(img_path, target_size=(224, 224)):
#     """
#     Preprocesses an image for prediction.

#     Args:
#         img_path (str): Path to the image file.
#         target_size (tuple): Target size for resizing the image (height, width).

#     Returns:
#         np.array: Preprocessed image as a numpy array.
#     """
#     # Load the image and resize it to the target size
#     img = image.load_img(img_path, target_size=target_size)
    
#     # Convert the image to a numpy array
#     img_array = image.img_to_array(img)
    
#     # Expand dimensions to match the model's input shape (batch size of 1)
#     img_array = np.expand_dims(img_array, axis=0)
    
#     # Normalize pixel values to the range [0, 1]
#     img_array /= 255.0
    
#     return img_array

#ChatGPT code starts

# import numpy as np
# from tensorflow.keras.preprocessing import image
# from PIL import Image

# def preprocess_image(img_path, target_size=(224, 224)):
#     """
#     Preprocesses an image for prediction.

#     Args:
#         img_path (str): Path to the image file.
#         target_size (tuple): Target size for resizing the image (height, width).

#     Returns:
#         np.array: Preprocessed image as a numpy array.
#     """
#     # Load the image, convert to RGB to ensure consistency
#     img = Image.open(img_path).convert("RGB")

#     # Resize the image
#     img = img.resize(target_size)

#     # Convert the image to a numpy array
#     img_array = np.array(img, dtype=np.float32)  # Ensure float32 for TensorFlow compatibility

#     # Normalize pixel values to the range [0, 1]
#     img_array /= 255.0

#     # Expand dimensions to match the model's expected input shape
#     img_array = np.expand_dims(img_array, axis=0)

#     return img_array

import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

def preprocess_image(img_path, target_size=(224, 224)):
    """
    Preprocesses an image for model prediction.

    Args:
        img_path (str): Path to the image file.
        target_size (tuple): Target size for resizing the image (height, width).

    Returns:
        np.array: Preprocessed image as a numpy array.
    """
    # Open image, ensure it's in RGB mode (not grayscale)
    img = Image.open(img_path).convert("RGB")

    # Resize image
    img = img.resize(target_size)

    # Convert image to array
    img_array = np.array(img, dtype=np.float32)

    # Debugging: Print shape and pixel range
    print(f"🖼 Image Shape: {img_array.shape} | Min: {img_array.min()} | Max: {img_array.max()}")

    # Normalize pixels
    img_array /= 255.0

    # Expand dimensions to fit model input
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
