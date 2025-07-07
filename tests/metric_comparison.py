import os
import time
import cv2
import torch
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt



from app.api import encoded_images
from app.image_processing import imageEncoder, \
    euclidean_distance, minkowski_distance, \
    hamming_distance, jaccard_similarity, \
    to_binary_vector, pearson_correlation

from sentence_transformers import util


def plot_results(metrics_data, total_images, avg_prep_time):

    metric_names = list(metrics_data.keys())
    accuracies = [(data['correct_predictions'] / total_images) * 100 for data in metrics_data.values()]
    avg_metric_times = [data['metric_time'] / total_images for data in metrics_data.values()]


    plt.figure(figsize=(12, 7))
    bars = plt.bar(metric_names, accuracies, color='skyblue')
    plt.xlabel('Metrikler', fontsize=12)
    plt.ylabel('Doğruluk (%)', fontsize=12)
    plt.title('Logo Tanıma Metriklerinin Doğruluk Karşılaştırması', fontsize=14, fontweight='bold')
    plt.ylim(0, 105)
    plt.xticks(rotation=45, ha='right')


    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    #plt.savefig('dogruluk_raporu.png')
    plt.show()


    prep_times = [avg_prep_time] * len(metric_names)

    plt.figure(figsize=(12, 7))
    plt.bar(metric_names, prep_times, label='Ort. Resim Hazırlama Süresi', color='orange')
    plt.bar(metric_names, avg_metric_times, bottom=prep_times, label='Ort. Metrik Kıyaslama Süresi', color='teal')

    plt.xlabel('Metrikler', fontsize=12)
    plt.ylabel('Ortalama Süre (saniye/resim)', fontsize=12)
    plt.title('Logo Tanıma Metriklerinin Ortalama İşlem Süreleri', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    #plt.savefig('zaman_raporu.png')
    plt.show()


def test_logo_recognition():
    test_images_dir = "./test_images"
    if not os.path.exists(test_images_dir):
        print(f"HATA: '{test_images_dir}' dizini bulunamadı.")
        print("Lütfen test resimlerini bu dizine 'marka_ismi-sayı.jpg' (örn: 'ADIDAS-1.jpg') formatında yerleştirin.")
        return

    # Her bir metrik için performans verilerini tutacak dict
    metrics = {
        "Cosine Similarity": {
            "best_score_initial": -float("inf"),
            "comparison_func": lambda s, bs: s > bs,
            "metric_time": 0.0,
            "correct_predictions": 0,
            "unit": "benzerlik (yüksek daha iyi)"
        },
        "Euclidean Distance": {
            "best_score_initial": float("inf"),
            "comparison_func": lambda s, bs: s < bs,
            "metric_time": 0.0,
            "correct_predictions": 0,
            "unit": "uzaklık (düşük daha iyi)"
        },
        "Hamming Distance": {
            "best_score_initial": float("inf"),
            "comparison_func": lambda s, bs: s < bs,
            "metric_time": 0.0,
            "correct_predictions": 0,
            "unit": "uzaklık (düşük daha iyi)"
        },
        "Dot Product": {
            "best_score_initial": -float("inf"),
            "comparison_func": lambda s, bs: s > bs,
            "metric_time": 0.0,
            "correct_predictions": 0,
            "unit": "benzerlik (yüksek daha iyi)"
        },
        "Chebyshev Distance (L_inf)": {
            "best_score_initial": float("inf"),
            "comparison_func": lambda s, bs: s < bs,
            "metric_time": 0.0,
            "correct_predictions": 0,
            "unit": "uzaklık (düşük daha iyi)"
        },
        "Minkowski Distance (p=3)": {
            "best_score_initial": float("inf"),
            "comparison_func": lambda s, bs: s < bs,
            "metric_time": 0.0,
            "correct_predictions": 0,
            "unit": "uzaklık (düşük daha iyi)"
        },
        "Jaccard Similarity": {
            "best_score_initial": -float("inf"),
            "comparison_func": lambda s, bs: s > bs,
            "metric_time": 0.0,
            "correct_predictions": 0,
            "unit": "benzerlik (yüksek daha iyi)"
        },
        "Pearson Correlation": {
            "best_score_initial": -float("inf"),
            "comparison_func": lambda s, bs: s > bs,
            "metric_time": 0.0,
            "correct_predictions": 0,
            "unit": "benzerlik (yüksek daha iyi)"
        }
    }

    total_test_images_processed = 0
    total_prep_time = 0
    for filename in os.listdir(test_images_dir):
        try:
            true_logo = filename.split("-")[0]
            if true_logo not in encoded_images:
                print(f"UYARI: '{true_logo}' için kayıtlı embedding bulunamadı. '{filename}' atlanıyor.")
                continue

            image_path = os.path.join(test_images_dir, filename)
            rawImage = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            if rawImage is None:
                print(f"UYARI: '{filename}' dosyası okunamadı veya bozuk. Atlanıyor.")
                continue

            total_test_images_processed += 1
            print(f"\n{'=' * 40}")
            print(f"Resim işleniyor: {filename} (Marka: {true_logo})")

            start_image_prep = time.perf_counter()
            image_embedding = imageEncoder(rawImage)
            image_np = image_embedding.squeeze(0).detach().numpy()
            end_image_prep = time.perf_counter()
            image_prep_time = end_image_prep - start_image_prep
            total_prep_time += image_prep_time
            print(f"  Resim okuma ve embedding çıkarma süresi: {image_prep_time:.4f} saniye")

            for metric_name, data in metrics.items():
                current_best_score = data["best_score_initial"]
                current_best_logo = None

                start_metric_calc_time = time.perf_counter()
                all_logo_scores = {}

                for logo, encoded_logo in encoded_images.items():
                    encoded_logo_np = encoded_logo.squeeze(0).detach().numpy()
                    score = 0.0

                    if metric_name == "Cosine Similarity":
                        score = float(util.pytorch_cos_sim(image_embedding, encoded_logo)[0][0])
                    elif metric_name == "Euclidean Distance":
                        score = euclidean_distance(image_embedding, encoded_logo)
                    elif metric_name == "Hamming Distance":
                        score = hamming_distance(image_embedding, encoded_logo)
                    elif metric_name == "Dot Product":
                        score = torch.dot(image_embedding.squeeze(0), encoded_logo.squeeze(0)).item()
                    elif metric_name == "Chebyshev Distance (L_inf)":
                        score = distance.chebyshev(image_np, encoded_logo_np)
                    elif metric_name == "Minkowski Distance (p=3)":
                        score = minkowski_distance(image_embedding, encoded_logo, p=3)
                    elif metric_name == "Jaccard Similarity":
                        binary_img = to_binary_vector(image_embedding)
                        binary_encoded_logo = to_binary_vector(encoded_logo)
                        intersection = (binary_img * binary_encoded_logo).sum().item()
                        union = (binary_img + binary_encoded_logo).sum().item()
                        score = intersection / union if union != 0 else 0
                    elif metric_name == "Pearson Correlation":
                        score = pearson_correlation(image_embedding, encoded_logo)

                    all_logo_scores[logo] = score

                    if data["comparison_func"](score, current_best_score):
                        current_best_score = score
                        current_best_logo = logo

                score_for_original_logo = all_logo_scores.get(true_logo)
                end_metric_calc_time = time.perf_counter()

                metrics[metric_name]["metric_time"] += (end_metric_calc_time - start_metric_calc_time)

                is_correct = (current_best_logo == true_logo)
                if is_correct:
                    metrics[metric_name]["correct_predictions"] += 1

                print(f"  [{metric_name}]")
                print(f"    Tahmin Edilen Logo: {current_best_logo} (Skor: {current_best_score:.4f}, {data['unit']})")
                print(f"    Doğruluk: {'DOĞRU' if is_correct else 'YANLIŞ'}")
                if score_for_original_logo is not None:
                    print(f"    Gerçek Logo ({true_logo}) Skoru: {score_for_original_logo:.4f}")
                print(f"    Metriğin kıyaslama süresi: {end_metric_calc_time - start_metric_calc_time:.4f} saniye")


        except Exception as e:
            print(f"HATA işlenirken '{filename}': {e}")
            raise e

    print(f"\n{'=' * 40}")

    if total_test_images_processed == 0:
        print("Hiç test resmi işlenemedi. Raporlama yapılamıyor.")
        return

    avg_ptime_per_image = total_prep_time / total_test_images_processed

    print("GENEL PERFORMANS RAPORU")
    print(f"Toplam başarılı şekilde işlenen test resmi: {total_test_images_processed}")
    print(f"Ortalama resim prep süresi: {avg_ptime_per_image:.4f} saniye/resim")

    for metric_name, data in metrics.items():
        accuracy = (data["correct_predictions"] / total_test_images_processed) * 100
        avg_mtime_per_image = data["metric_time"] / total_test_images_processed
        avg_time_per_image = avg_ptime_per_image + avg_mtime_per_image

        print(f"\nMetrik: {metric_name}")
        print(f"  Doğruluk: {data['correct_predictions']}/{total_test_images_processed} ({accuracy:.2f}%)")
        print(
            f"  Ortalama Kıyaslama Süresi: {avg_mtime_per_image:.4f} saniye/resim")
        print(f"  Ortalama Toplam Süre (Hazırlama+Kıyaslama): {avg_time_per_image:.4f} saniye/resim")


    print(f"\n{'=' * 40}")
    print("Grafik oluşturuluyor...")
    plot_results(metrics, total_test_images_processed, avg_ptime_per_image)


if __name__ == '__main__':
    test_logo_recognition()