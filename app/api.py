import os
import torch
import open_clip
from sentence_transformers import util
from PIL import Image
import cv2
import easyocr
import numpy as np
from flask import request, jsonify

from app.image_processing import generateScore, saveEncodedImages, loadEncodedImages, get_color_name

# from .image_processing import generateScore, loadEncodedImages, saveEncodedImages # Assuming you move saveEncodedImages here or import it from your first script
# Assuming the first script is named 'image_encoding_utils.py'


# --- Configuration ---
logo_dataset_path = 'logo_dataset/'
encoded_images_path = 'encoded_images_l/'

# --- Initial Setup for Encoding Images (Run Once) ---
# Ensure the encoded_images directory exists
if not os.path.exists(encoded_images_path):
    os.makedirs(encoded_images_path)
    print(f"Created directory: {encoded_images_path}")

# Check if encoded images already exist to avoid re-encoding every time
# You might want a more robust check (e.g., comparing timestamps or checksums)
# if you expect the dataset to change frequently.
if not os.listdir(encoded_images_path):
    print("No encoded images found. Encoding and saving images...")
    saveEncodedImages(logo_dataset_path, encoded_images_path)
    print("Image encoding complete.")
else:
    print("Encoded images already exist. Skipping encoding process.")

# --- Load Encoded Images for Application Use ---
encoded_images = loadEncodedImages(encoded_images_path)
print(f"Loaded {len(encoded_images)} encoded images.")

# Initialize EasyOCR reader globally
reader = easyocr.Reader(['en'])

# --- Helper Functions (if moved from the first script or if you want to keep them here) ---
def get_color_around_text(image, box):
    """
    Returns the average BGR color of pixels around the corners of the bounding box.
    """
    colors = []
    # Ensure coordinates are within image bounds
    for (x, y) in box:
        x, y = int(x), int(y)
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            colors.append(image[y, x])
    if not colors:
        return None
    avg_color = np.mean(colors, axis=0)
    return avg_color.tolist() # Return as a list of floats (BGR)

def detect_column_code_and_color(image):
    """
    Detects text (column code) and its surrounding color using EasyOCR.
    """
    result = reader.readtext(image)

    detected_text_parts = []
    avg_color = None

    for detection in result:
        text = detection[1]
        box = detection[0]
        # Filter for alphanumeric characters if you only care about codes
        detected_text_parts.append(''.join(filter(str.isalnum, text)))

        # Get color from the first detected text bounding box
        if avg_color is None:
            color = get_color_around_text(image, box)
            if color:
                avg_color = color

    combined_text = ''.join(detected_text_parts)
    return combined_text, avg_color


def register_routes(app):
    """
    Registers Flask routes for image processing, including
    separate endpoints for logo detection and OCR.
    """

    @app.route('/detect-logo', methods=['POST'])
    def detect_logo():
        """
        Endpoint for logo detection.
        Expects an image file in the request.
        """
        if 'image' not in request.files:
            return jsonify({'error': 'Resim dosyası bulunamadı'}), 400

        image_file = request.files['image']
        image_pil = Image.open(image_file)
        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        try:
            best_match, best_score, best_method = generateScore(image_cv2, encoded_images)

            if best_score >= 0.4:  # Your defined confidence threshold
                response = {
                    'message': 'Logo eşleşti.',
                    'best_match': best_match,
                    'score': best_score,
                    'method': best_method
                }
                return jsonify(response), 200
            else:
                return jsonify({
                    'message': 'Logo eşleşmesi bulunamadı veya güven eşiğinin altında.',
                    'best_match': None,
                    'score': best_score
                }), 404

        except Exception as e:
            print(f"Logo algılama hatası: {e}")
            return jsonify({'error': f"Resim işlenirken bir hata oluştu: {str(e)}"}), 500

    @app.route('/perform-ocr', methods=['POST'])
    def perform_ocr():
        """
        Endpoint for OCR (Optical Character Recognition).
        Expects an image file in the request.
        """
        if 'image' not in request.files:
            return jsonify({'error': 'Resim dosyası bulunamadı'}), 400

        image_file = request.files['image']
        image_pil = Image.open(image_file)
        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        try:
            code, color_bgr = detect_column_code_and_color(image_cv2)

            if not code and color_bgr is None:
                return jsonify({'message': 'Kolon kodu veya renk tespit edilemedi.'}), 404

            color_name = None
            if color_bgr:
                color_name = get_color_name(color_bgr) # Assuming get_color_name is available

            response = {
                'message': 'OCR başarıyla tamamlandı.',
                'column_code': code,
                'column_color_bgr': color_bgr,
                'column_color_name': color_name
            }
            return jsonify(response), 200

        except Exception as e:
            print(f"OCR hatası: {e}")
            return jsonify({'error': f"Resim işlenirken bir hata oluştu: {str(e)}"}), 500
