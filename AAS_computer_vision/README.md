# OCR Plat Nomor Kendaraan menggunakan Visual Language Model (LM Studio)

Tugas UTS Computer Vision (RE604) — OCR plat nomor kendaraan Indonesia
menggunakan Visual Language Model (VLM) yang dijalankan melalui **LM Studio**
dan diintegrasikan dengan Python, dievaluasi dengan **Character Error Rate (CER)**.

## Struktur Project

```
plate-ocr-vlm/
├── main.py              # Script utama: loop dataset -> panggil VLM -> hitung CER -> simpan CSV
├── lmstudio_client.py    # Wrapper untuk memanggil LM Studio local server (OpenAI-compatible API)
├── dataset_utils.py      # Loader gambar + ground truth dari folder test
├── cer_metric.py         # Implementasi metrik CER (S, D, I, N) dari nol
├── requirements.txt
└── README.md
```

## 1. Persiapan LM Studio

1. Download & install LM Studio dari https://lmstudio.ai
2. Buka LM Studio, masuk ke tab **Discover / Search**, cari model multimodal
   (vision) misalnya salah satu dari:
   - `llava-v1.5-7b`
   - `bakllava-1`
   - `qwen2-vl-7b-instruct`
   - `moondream2` (lebih ringan, cocok untuk laptop dengan RAM/VRAM terbatas)
3. Download model tersebut (pilih quantization sesuai RAM/VRAM, misalnya Q4_K_M).
4. Buka tab **Local Server** (ikon `<->` di sidebar kiri).
5. Pilih model vision yang sudah didownload pada dropdown model.
6. Klik **Start Server**. Secara default server berjalan di:
   `http://localhost:1234/v1`
7. Pastikan server menunjukkan status running dan siap menerima request
   `/v1/chat/completions`.

Referensi resmi: https://lmstudio.ai/docs/python/llm-prediction/image-input

## 2. Persiapan Dataset

1. Download dataset dari Kaggle:
   https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset
2. Ekstrak dataset, gunakan folder **test** saja sesuai instruksi soal.
3. **Penting:** cek isi folder `test` terlebih dahulu:
   - Jika ada file anotasi (`_annotations.csv`, `annotations.csv`, atau `labels.csv`)
     berisi kolom nama file dan teks plat, script akan otomatis membacanya.
   - Jika tidak ada file anotasi, script akan menggunakan **nama file gambar**
     sebagai ground truth (mis. `B1234XYZ.jpg` -> ground truth `B1234XYZ`).
     Jika konvensi penamaan dataset Anda berbeda, sesuaikan fungsi
     `load_ground_truth_map()` di `dataset_utils.py`.

## 3. Instalasi Dependency

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Menjalankan Program

```bash
python main.py --data-dir "path/ke/dataset/test" --output results.csv
```

Argumen opsional:
- `--base-url` : URL LM Studio local server (default `http://localhost:1234/v1`)
- `--model`    : nama model (opsional, biasanya tidak perlu diisi karena LM Studio
                 otomatis memakai model yang sedang di-load)
- `--limit`    : batasi jumlah gambar yang diproses, berguna untuk uji coba cepat,
                 contoh `--limit 10`

Contoh uji coba cepat dengan 10 gambar pertama:
```bash
python main.py --data-dir "./dataset/test" --output results_sample.csv --limit 10
```

## 5. Output

Program menghasilkan file CSV (`results.csv`) dengan kolom:

| image | ground_truth | prediction | CER_score |
|-------|--------------|------------|-----------|

Ringkasan (jumlah data & rata-rata CER) akan ditampilkan di terminal setelah
proses selesai.

## 6. Rumus CER yang digunakan

```
CER = (S + D + I) / N
```
- S = jumlah karakter substitusi
- D = jumlah karakter dihapus (ada di ground truth, tidak muncul di prediksi)
- I = jumlah karakter disisipkan (muncul di prediksi, tidak ada di ground truth)
- N = jumlah karakter pada ground truth

Dihitung menggunakan algoritma Levenshtein distance dengan backtrace pada
`cer_metric.py`, sehingga S, D, I dipisahkan secara eksplisit, bukan hanya
total edit distance.

## Catatan

- Prompt yang dikirim ke model VLM:
  `"What is the license plate number shown in this image? Respond only with the plate number."`
- Jika hasil prediksi model masih mengandung teks tambahan (misalnya
  "The plate number is: ..."), fungsi `_clean_prediction()` di
  `lmstudio_client.py` akan membersihkannya secara sederhana. Silakan
  sesuaikan lagi jika model yang dipakai punya gaya jawaban berbeda.
