"""
cer_metric.py

Implementasi Character Error Rate (CER) sesuai formula pada soal:

    CER = (S + D + I) / N

    S = jumlah karakter substitusi
    D = jumlah karakter dihapus (ada di ground truth, hilang di prediksi)
    I = jumlah karakter disisipkan (ada di prediksi, tidak ada di ground truth)
    N = jumlah karakter pada ground truth

Dihitung menggunakan algoritma Levenshtein distance (dynamic programming)
dengan backtrace, sehingga S, D, I bisa dipisahkan (bukan cuma total edit
distance).
"""

def _normalize(text: str) -> str:
    """Normalisasi teks plat nomor sebelum dibandingkan.
    Uppercase + hilangkan spasi, karena LMStudio/VLM kadang menambahkan
    spasi atau huruf kecil pada hasil prediksi.
    """
    if text is None:
        return ""
    return "".join(text.upper().split())


def compute_cer(ground_truth: str, prediction: str):
    """Hitung CER antara ground_truth dan prediction.

    Returns
    -------
    dict berisi: cer, S, D, I, N
    """
    ref = _normalize(ground_truth)
    hyp = _normalize(prediction)

    n, m = len(ref), len(hyp)

    # dp[i][j] = biaya minimum edit distance antara ref[:i] dan hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i  # i penghapusan
    for j in range(m + 1):
        dp[0][j] = j  # j penyisipan

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitution = dp[i - 1][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                insertion = dp[i][j - 1] + 1
                dp[i][j] = min(substitution, deletion, insertion)

    # Backtrace untuk memisahkan jumlah S, D, I
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
            continue
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            I += 1
            j -= 1
        else:
            # fallback pengaman, seharusnya tidak pernah tercapai
            break

    N = n if n > 0 else 1  # hindari pembagian dengan nol
    cer = (S + D + I) / N

    return {"cer": round(cer, 4), "S": S, "D": D, "I": I, "N": n}


if __name__ == "__main__":
    # Contoh cepat untuk verifikasi manual
    examples = [
        ("BP1234CD", "BP1234CD"),   # identik -> CER 0
        ("BP1234CD", "BP1284CD"),   # 1 substitusi
        ("BP1234CD", "BP123CD"),    # 1 penghapusan
        ("BP1234CD", "BP12334CD"),  # 1 penyisipan
    ]
    for gt, pred in examples:
        print(gt, "|", pred, "->", compute_cer(gt, pred))
