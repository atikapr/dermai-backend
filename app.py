from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
import json

app = Flask(__name__)

# ==========================================
# 1. LOAD ARTEFAK & MODEL (LATE FUSION)
# ==========================================

# A. Load Model Citra (TFLite - 14.88 MB)
TFLITE_MODEL_PATH = "skin_cancer_image_model.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# B. Load Model Metadata (Keras H5 - MLP)
# Gunakan compile=False agar Flask tidak bingung mencari fungsi FocalLoss
MLP_MODEL_PATH = "best_mlp_model.tflite"
mlp_model = load_model(MLP_MODEL_PATH, compile=False)

# C. Load Meta Info (Agar batas umur dinamis sesuai dataset Colab)
try:
    with open("meta_info.json", "r") as f:
        meta_info = json.load(f)
        AGE_MIN = meta_info["age_min"]
        AGE_MAX = meta_info["age_max"]
except FileNotFoundError:
    # Fallback jika file json lupa di-upload
    AGE_MIN, AGE_MAX = 5.0, 85.0

# Daftar kelas penyakit (Pastikan urutan sama persis dengan Colab)
CLASS_NAMES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "SCCKA", "VASC"]

# Bobot Optimal Hasil Grid Search (Colab Bagian 10)
ALPHA = 0.60  # Bobot Citra
BETA = 0.40  # Bobot Metadata

# ==========================================
# 2. ENDPOINT API
# ==========================================


@app.route("/")
def home():
    return "Server DermAI Multimodal Aktif dan Berjalan Mulus!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # A. TANGKAP DATA DARI ANDROID
        if "image" not in request.files:
            return jsonify({"error": "Tidak ada gambar yang dikirim"}), 400

        file = request.files["image"]
        age = float(request.form.get("age", 0))
        sex = int(request.form.get("sex", 0))  # 0: Female, 1: Male
        lokasi = request.form.get("lokasi", "head/neck").lower().strip()

        # B. PRA-PEMROSESAN CITRA
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

        # C. PRA-PEMROSESAN METADATA (6 Fitur)
        # 1. Normalisasi umur berdasarkan dataset riil
        age_norm = (age - AGE_MIN) / (AGE_MAX - AGE_MIN)
        age_norm = max(0.0, min(1.0, age_norm))

        # 2. Encoding Gender (Male=1.0, Female=0.0)
        sex_enc = float(sex)

        # 3. One-Hot Encoding Lokasi Lesi ('trunk' jadi 0 semua)
        loc_head = 1.0 if lokasi == "head/neck" else 0.0
        loc_lower = 1.0 if lokasi == "lower extremity" else 0.0
        loc_oral = 1.0 if lokasi == "oral/genital" else 0.0
        loc_upper = 1.0 if lokasi == "upper extremity" else 0.0

        meta_array = np.array(
            [[age_norm, sex_enc, loc_head, loc_lower, loc_oral, loc_upper]],
            dtype=np.float32,
        )  # Shape: (1, 6)

        # ==========================================
        # 3. PROSES LATE FUSION PREDICTION
        # ==========================================

        # Langkah 1: Prediksi Citra (TFLite)
        interpreter.set_tensor(input_details[0]["index"], img_array)
        interpreter.invoke()
        prob_image = interpreter.get_tensor(output_details[0]["index"])[
            0
        ]  # Array 8 probabilitas

        # Langkah 2: Prediksi Metadata (Keras MLP)
        prob_meta = mlp_model.predict(meta_array, verbose=0)[0]  # Array 8 probabilitas

        # Langkah 3: Penggabungan (Late Fusion) dengan Bobot Alpha & Beta
        prob_fused = (ALPHA * prob_image) + (BETA * prob_meta)

        # Langkah 4: Ambil Keputusan Akhir
        max_index = int(np.argmax(prob_fused))
        predicted_class = CLASS_NAMES[max_index]
        confidence = float(prob_fused[max_index]) * 100

        # Kirim response ke Android
        return jsonify(
            {
                "status": "success",
                "diagnosis": predicted_class,
                "confidence": f"{confidence:.2f}%",
                "image_weight": f"{ALPHA*100}%",
                "clinical_weight": f"{BETA*100}%",
                "message": "Diagnosis Multimodal berhasil dilakukan",
            }
        )

    except Exception as e:
        print("\n====== ALARM ERROR AI ======")
        print(str(e))
        print("============================\n")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
