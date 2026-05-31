from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import json

app = Flask(__name__)

# ==========================================
# 1. LOAD ARTEFAK & 2 MODEL (FULL TFLITE - SUPER RINGAN)
# ==========================================

# A. Load Model Citra (TFLite)
img_interpreter = tf.lite.Interpreter(model_path="skin_cancer_image_model.tflite")
img_interpreter.allocate_tensors()
img_input_details = img_interpreter.get_input_details()
img_output_details = img_interpreter.get_output_details()

# B. Load Model Metadata (TFLite)
mlp_interpreter = tf.lite.Interpreter(model_path="best_mlp_model.tflite")
mlp_interpreter.allocate_tensors()
mlp_input_details = mlp_interpreter.get_input_details()
mlp_output_details = mlp_interpreter.get_output_details()

# C. Load Meta Info (Untuk Normalisasi Umur)
try:
    with open("meta_info.json", "r") as f:
        meta_info = json.load(f)
        AGE_MIN = meta_info["age_min"]
        AGE_MAX = meta_info["age_max"]
except FileNotFoundError:
    AGE_MIN, AGE_MAX = 5.0, 85.0

CLASS_NAMES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "SCCKA", "VASC"]

# Bobot Optimal Fusi
ALPHA = 0.60
BETA = 0.40

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
        sex = int(request.form.get("sex", 0))
        lokasi = request.form.get("lokasi", "head/neck").lower().strip()

        # B. PRA-PEMROSESAN CITRA
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # C. PRA-PEMROSESAN METADATA (6 Fitur)
        age_norm = (age - AGE_MIN) / (AGE_MAX - AGE_MIN)
        age_norm = max(0.0, min(1.0, age_norm))
        sex_enc = float(sex)

        loc_head = 1.0 if lokasi == "head/neck" else 0.0
        loc_lower = 1.0 if lokasi == "lower extremity" else 0.0
        loc_oral = 1.0 if lokasi == "oral/genital" else 0.0
        loc_upper = 1.0 if lokasi == "upper extremity" else 0.0

        meta_array = np.array(
            [[age_norm, sex_enc, loc_head, loc_lower, loc_oral, loc_upper]],
            dtype=np.float32,
        )

        # ==========================================
        # 3. PROSES LATE FUSION PREDICTION (Keduanya pakai TFLite)
        # ==========================================

        # Prediksi Citra (Interpreter 1)
        img_interpreter.set_tensor(img_input_details[0]["index"], img_array)
        img_interpreter.invoke()
        prob_image = img_interpreter.get_tensor(img_output_details[0]["index"])[0]

        # Prediksi Metadata (Interpreter 2)
        mlp_interpreter.set_tensor(mlp_input_details[0]["index"], meta_array)
        mlp_interpreter.invoke()
        prob_meta = mlp_interpreter.get_tensor(mlp_output_details[0]["index"])[0]

        # Penggabungan (Late Fusion)
        prob_fused = (ALPHA * prob_image) + (BETA * prob_meta)

        # Keputusan Akhir
        max_index = int(np.argmax(prob_fused))
        predicted_class = CLASS_NAMES[max_index]
        confidence = float(prob_fused[max_index]) * 100

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
