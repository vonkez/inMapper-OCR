import os

from flask import request, jsonify
from .image_processing import generateScore, loadEncodedImages, saveEncodedImages
from PIL import Image
import numpy as np
import cv2

# Encode edilmiş logo resimlerini yükle
# Encode edilmiş resimleri kaydet
logo_dataset_path = 'logo_dataset/'  # Logo dataset dizini
encoded_images_path = 'encoded_images/'  # Encode edilmiş resimlerin bulunduğu dizin
#if not os.path.exists(encoded_images_path):
#    os.makedirs(encoded_images_path)
#saveEncodedImages(logo_dataset_path, encoded_images_path)
encoded_images = loadEncodedImages(encoded_images_path)

# Endpoint: Resim yükleme ve benzerlik skoru döndürme
def register_routes(app):
    @app.route('/compare-logo', methods=['POST'])
    def compare_logo():
        if 'image' not in request.files:
            return jsonify({'error': 'Resim dosyası bulunamadı'}), 400

        # Resmi oku
        image_file = request.files['image']
        image = Image.open(image_file)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        try:
            # Benzerlik skorlarını hesapla
            best_match, best_score, best_method = generateScore(image, encoded_images)

            return jsonify({
                'best_match': best_match,
                'score': best_score,
                'method': best_method
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500
