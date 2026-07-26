"""
main.py

Script utama tugas: OCR plat nomor kendaraan menggunakan Visual Language
Model (VLM) yang dijalankan lewat LM Studio, dievaluasi dengan Character
Error Rate (CER).

Cara pakai:
    python main.py --data-dir "./dataset/test" --output results.csv

Prasyarat sebelum menjalankan:
    1. LM Studio sudah terbuka, model vision (llava/bakllava/qwen2-vl/dll)
       sudah di-load, dan Local Server sudah di-Start (default port 1234).
    2. Dataset sudah didownload dari Kaggle dan folder "test" sudah di-ekstrak.
    3. Install dependency: pip install -r requirements.txt
"""

import argparse
import csv
import os
import time

from lmstudio_client import LMStudioVLMClient
from dataset_utils import load_ground_truth_map, list_test_images
from cer_metric import compute_cer

PROMPT = "What is the license plate number shown in this image? Respond only with the plate number."


def main():
    parser = argparse.ArgumentParser(description="OCR plat nomor kendaraan pakai VLM (LM Studio)")
    parser.add_argument("--data-dir", required=True, help="Path ke folder test dataset")
    parser.add_argument("--output", default="results.csv", help="Path file CSV hasil")
    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="URL LM Studio local server")
    parser.add_argument("--model", default=None, help="Nama model (opsional, biasanya tidak wajib diisi)")
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah gambar yang diproses (untuk testing cepat)")
    args = parser.parse_args()

    client = LMStudioVLMClient(base_url=args.base_url, model=args.model)

    print(f"[INFO] Membaca ground truth dari: {args.data_dir}")
    gt_map = load_ground_truth_map(args.data_dir)
    image_paths = list_test_images(args.data_dir)
    if args.limit:
        image_paths = image_paths[: args.limit]

    print(f"[INFO] Total gambar yang akan diproses: {len(image_paths)}")

    rows = []
    total_cer = 0.0

    for idx, img_path in enumerate(image_paths, start=1):
        # Menggunakan os.path.basename agar aman cross-platform (Windows/Linux/Mac)
        fname = os.path.basename(img_path)
        ground_truth = gt_map.get(fname, "")

        try:
            t0 = time.time()
            raw_prediction = client.predict_plate(img_path, prompt=PROMPT)
            elapsed = time.time() - t0

            # Pembersihan whitespace & tanda kutip bawaan VLM
            prediction = raw_prediction.strip().strip('"\'') if raw_prediction else ""
        except Exception as e:
            print(f"[ERROR] Gagal memproses {fname}: {e}")
            prediction = ""
            elapsed = 0.0

        metrics = compute_cer(ground_truth, prediction)
        cer_score = metrics["cer"]
        total_cer += cer_score

        rows.append({
            "image": fname,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "CER_score": cer_score,
        })

        print(f"[{idx}/{len(image_paths)}] {fname} | GT: {ground_truth} | Pred: {prediction} "
              f"| CER: {cer_score:.4f} | {elapsed:.1f}s")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "ground_truth", "prediction", "CER_score"])
        writer.writeheader()
        writer.writerows(rows)

    avg_cer = total_cer / len(rows) if rows else 0.0
    print("\n=== RINGKASAN ===")
    print(f"Jumlah data      : {len(rows)}")
    print(f"Rata-rata CER    : {avg_cer:.4f}")
    print(f"Hasil disimpan ke: {args.output}")


if __name__ == "__main__":
    main()