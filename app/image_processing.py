import os
import torch
import open_clip
from sentence_transformers import util
from PIL import Image
import cv2
# Cihazı belirle
device = "cuda" if torch.cuda.is_available() else "cpu"

# OpenCLIP modelini ve ön işleme adımlarını oluştur
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
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
    return (vec1 != vec2).float().sum().item() / len(vec1)

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
    # Test resmini encodela
    img1 = imageEncoder(test_img)

    best_score = None
    best_logo = None
    best_method = None  # En yüksek skoru veren yöntemi tutacak değişken

    # Encode edilmiş logo resimleri ile karşılaştırma yap
    for logo, encoded_logo in encoded_images.items():
        # Kozinüs benzerliğini hesapla
        cos_scores = util.pytorch_cos_sim(img1, encoded_logo)
        cos_score = float(cos_scores[0][0])

        # Öklidyen mesafeyi hesapla (negatif değer, benzerliği artırır)
        euclidean_dist = euclidean_distance(img1, encoded_logo)

        # Minkowski mesafesini hesapla (p=3 örneği)
        minkowski_dist = minkowski_distance(img1, encoded_logo, p=3)

        # İkili vektörleri oluştur
        binary_img1 = to_binary_vector(img1)
        binary_encoded_logo = to_binary_vector(encoded_logo)

        # Jaccard benzerliğini hesapla
        intersection = (binary_img1 * binary_encoded_logo).sum().item()
        union = (binary_img1 + binary_encoded_logo).sum().item()
        jaccard_score = intersection / union if union != 0 else 0

        # Pearson korelasyonunu hesapla
        pearson_corr = pearson_correlation(img1, encoded_logo)

        # Benzerlik skorlarını birleştir
        combined_score = {
            'cosine_similarity': cos_score,
            'euclidean_distance': -euclidean_dist,  # Negatif
            'minkowski_distance': -minkowski_dist,  # Negatif
            'jaccard_similarity': jaccard_score,
            'pearson_correlation': pearson_corr
        }

        # En yüksek benzerliği kontrol et
        current_best = max(combined_score.values())
        current_best_method = max(combined_score, key=combined_score.get)  # En yüksek skoru veren yöntem

        if best_score is None or current_best > best_score:
            best_score = current_best
            best_logo = logo
            best_method = current_best_method  # En iyi yöntemi güncelle

    return best_logo, best_score, best_method  # En iyi yöntemi döndür