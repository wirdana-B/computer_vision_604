"""
dataset_utils.py

Utility untuk membaca folder `test` dari dataset
"Indonesian License Plate Recognition" dan mengambil pasangan
(path_gambar, ground_truth_text).
"""

import os
import re
import csv
import json

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def _clean_stem_as_plate(stem: str) -> str:
    """Ubah nama file (tanpa ekstensi) menjadi kandidat teks plat nomor."""
    stem = re.sub(r"[\(\)\[\]]", "", stem)
    stem = re.sub(r"[-_]?copy.*$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[-_]?\d+$", "", stem) if re.search(r"[-_]\d+$", stem) else stem
    stem = stem.replace("_", "").replace("-", "").replace(" ", "")
    return stem.upper()


def _try_load_txt_label_dir(test_dir: str, fname: str) -> str:
    """Cari file .txt pasangan gambar di folder labelswithLP/test."""
    stem = os.path.splitext(fname)[0]
    txt_filename = stem + ".txt"
    
    norm_test_dir = os.path.normpath(test_dir)
    
    # Menghubungkan test/images/test ke test/labelswithLP/test
    possible_dirs = [
        os.path.abspath(os.path.join(norm_test_dir, "..", "..", "labelswithLP", "test")),
        os.path.abspath(os.path.join(norm_test_dir, "..", "..", "labelswithLP")),
        os.path.abspath(os.path.join(norm_test_dir, "..", "labelswithLP")),
        norm_test_dir
    ]
    
    for pdir in possible_dirs:
        txt_path = os.path.join(pdir, txt_filename)
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                lines = content.splitlines()
                
                plates = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        # Mengambil elemen terakhir jika formatnya (class x y w h PLATE)
                        plate_text = parts[-1]
                        # Pastikan bukan sekadar angka float/koordinat
                        if not plate_text.replace('.', '', 1).isdigit():
                            plates.append(plate_text)
                        elif len(parts) == 1:
                            plates.append(parts[0])
                            
                if plates:
                    # Gabungkan dengan spasi jika ada lebih dari 1 plat di 1 gambar
                    return " ".join(plates)
                    
    return ""


def _try_load_annotation_file(test_dir: str):
    """Cari file anotasi CSV/JSON umum di dalam folder test."""
    candidates = ["_annotations.csv", "annotations.csv", "labels.csv", "_annotations.json", "labels.json"]
    for name in candidates:
        path = os.path.join(test_dir, name)
        if os.path.isfile(path):
            mapping = {}
            if path.endswith(".csv"):
                with open(path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    file_col = next((c for c in reader.fieldnames if c.lower() in ("filename", "image", "file")), None)
                    text_col = next((c for c in reader.fieldnames if c.lower() in ("text", "plate", "label", "ground_truth")), None)
                    if file_col and text_col:
                        for row in reader:
                            mapping[row[file_col]] = row[text_col]
            else:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        fname = item.get("filename") or item.get("image")
                        text = item.get("text") or item.get("plate") or item.get("label")
                        if fname and text:
                            mapping[fname] = text
            if mapping:
                return mapping
    return None


def load_ground_truth_map(test_dir: str) -> dict:
    """Kembalikan dict {nama_file_gambar: ground_truth_text}."""
    annotation_map = _try_load_annotation_file(test_dir)
    if annotation_map:
        return annotation_map

    mapping = {}
    for fname in os.listdir(test_dir):
        if fname.lower().endswith(IMAGE_EXTENSIONS):
            txt_gt = _try_load_txt_label_dir(test_dir, fname)
            if txt_gt:
                mapping[fname] = txt_gt
            else:
                stem = os.path.splitext(fname)[0]
                mapping[fname] = _clean_stem_as_plate(stem)
    return mapping


def list_test_images(test_dir: str):
    """Kembalikan list path lengkap semua gambar di folder test (urut nama)."""
    files = [f for f in os.listdir(test_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
    files.sort()
    return [os.path.join(test_dir, f) for f in files]