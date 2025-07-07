import os
import torch
import open_clip
from sentence_transformers import util
from PIL import Image
import cv2
import easyocr

# Cihazı belirle
device = "cuda" if torch.cuda.is_available() else "cpu"

# OpenCLIP modelini ve ön işleme adımlarını oluştur
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained="laion2b_s32b_b82k")
model.to(device)

def imageEncoder(img):
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1


def saveEncodedImages(logo_dataset_path, output_path):
    for filename in os.listdir(logo_dataset_path):
        logo_img_path = os.path.join(logo_dataset_path, filename)
        data_img = cv2.imread(logo_img_path, cv2.IMREAD_UNCHANGED)

        if data_img is None:
            print(f"Resim bulunamadı: {logo_img_path}")
            continue
        img_encoded = imageEncoder(data_img)

        filename_without_extension = os.path.splitext(filename)[0]
        output_filename = f"{filename_without_extension}.pt"  # Çıktı dosya adı
        output_file_path = os.path.join(output_path, output_filename)
        torch.save(img_encoded, output_file_path)

        print(f"{filename} encode edilmiş resim kaydedildi: {output_file_path}")

def loadEncodedImages(encoded_images_path):
    encoded_images = {}
    for filename in os.listdir(encoded_images_path):
        if filename.endswith('.pt'):
            file_path = os.path.join(encoded_images_path, filename)
            encoded_image = torch.load(file_path).to(device)
            encoded_images[os.path.splitext(filename)[0]] = encoded_image
    return encoded_images

def euclidean_distance(vec1, vec2):
    return torch.norm(vec1 - vec2).item()
def minkowski_distance(vec1, vec2, p):
    return torch.norm(vec1 - vec2, p=p).item()

def hamming_distance(vec1, vec2):
    return (~torch.isclose(vec1, vec2, atol=1e-1)).sum().item()

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def pearson_correlation(vec1, vec2):
    """Pearson korelasyonunu hesapla."""
    # Vektörleri düzleştir (flatten) ve iki boyutlu hale getir
    vec1 = vec1.view(-1)
    vec2 = vec2.view(-1)

    # Korelasyonu hesapla
    # detach() ile gradyan hesaplamasından ayırıyoruz
    corr_coef = torch.corrcoef(torch.stack((vec1.detach(), vec2.detach())))

    return float(corr_coef[0][1])  # [0][1] ile iki vektör arasındaki korelasyonu al
def to_binary_vector(vec, threshold=0.5):
    """Vektörü ikili (binary) vektöre dönüştür."""
    return (vec > threshold).float()

def generateScore(test_img, encoded_images):
    img1 = imageEncoder(test_img)

    best_combined_score = -float('inf') # Başlangıç değeri
    best_logo = None

    for logo, encoded_logo in encoded_images.items():
        # 1. Metrikleri Hesapla
        cos_scores = util.pytorch_cos_sim(img1, encoded_logo)
        cos_score = float(cos_scores[0][0])

        euclidean_dist = euclidean_distance(img1, encoded_logo)
        minkowski_dist = minkowski_distance(img1, encoded_logo, p=3)
        pearson_corr = pearson_correlation(img1, encoded_logo)

        # Jaccard için ikili vektör oluşturma
        binary_img1 = to_binary_vector(img1)
        binary_encoded_logo = to_binary_vector(encoded_logo)
        intersection = (binary_img1 * binary_encoded_logo).sum().item()
        union = (binary_img1 + binary_encoded_logo).sum().item()
        jaccard_score = intersection / union if union != 0 else 0

        # 2. Skorları Normalize Et (0-1 Aralığına)
        # Cosine ve Pearson zaten -1 ile 1 arasında, 0-1'e taşımak için (x + 1) / 2
        norm_cos_score = (cos_score + 1) / 2
        norm_pearson_corr = (pearson_corr + 1) / 2
        # Jaccard zaten 0-1 arasında

        # Mesafe ölçütlerini benzerliğe çevir
        # Mesafenin 0 olduğu durumda sonsuz benzerlik verir,
        # bu nedenle 1 / (1 + mesafe) formülü daha stabildir.
        # Maksimum olası mesafeyi bilmediğimiz için bu tür bir normalizasyon kullanıyoruz.
        norm_euclidean_score = 1 / (1 + euclidean_dist)
        norm_minkowski_score = 1 / (1 + minkowski_dist)

        weights = {
            'cosine': 0.1,
            'jaccard': 0.1,
            'pearson': 0.8
        }

        combined_score = (
            weights['cosine'] * norm_cos_score +
            weights['jaccard'] * jaccard_score +
            weights['pearson'] * norm_pearson_corr
        )

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_logo = logo

    return best_logo, best_combined_score, "combined_metrics"

def detect_column_code_and_color(img):

    reader = easyocr.Reader(['en', 'tr'], gpu=False)
    ocr_results = reader.readtext(img)
    code = None
    color_name = None

    for bbox, text, conf in ocr_results:
        if len(text) <= 5 and any(c.isalpha() for c in text) and any(c.isdigit() for c in text):
            code = text
            # bbox: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            # Kutuyu biraz büyüt (çevresini de al)
            pad_x = int((x_max - x_min) * 0.2)
            pad_y = int((y_max - y_min) * 0.2)
            x_min = max(0, x_min - pad_x)
            x_max = min(img.shape[1], x_max + pad_x)
            y_min = max(0, y_min - pad_y)
            y_max = min(img.shape[0], y_max + pad_y)

            roi = img[y_min:y_max, x_min:x_max]
            if roi.size > 0:
                avg_color = roi.mean(axis=(0, 1))  # BGR
                color_name = get_color_name(avg_color)
            else:
                color_name = None
            break

    return code, color_name

def get_color_name(bgr):
    """
    BGR renk değerini temel renge çevirir (ör: kırmızı, mavi, sarı, vs.)
    """
    colors = {
        'kırmızı': (0, 0, 255),
        'yeşil': (0, 255, 0),
        'mavi': (255, 0, 0),
        'sarı': (0, 255, 255),
        'turuncu': (0, 165, 255),
        'mor': (255, 0, 255),
        'siyah': (0, 0, 0),
        'beyaz': (255, 255, 255),
        'gri': (128, 128, 128)
    }
    min_dist = float('inf')
    closest_color = 'tanımsız'
    for name, value in colors.items():
        dist = sum((int(bgr[i]) - value[i]) ** 2 for i in range(3))
        if dist < min_dist:
            min_dist = dist
            closest_color = name
    return closest_color