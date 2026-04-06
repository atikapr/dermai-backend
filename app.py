from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = Flask(__name__)

# 1. MODEL TFLITE
MODEL_PATH = "Model_Hybrid_Tika.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Daftar kelas penyakit sesuai urutan Encoder di Colab
CLASS_NAMES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCCKA', 'VASC']

@app.route('/')
def home():
    return "Server DermAI Aktif dan Berjalan Mulus!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 2. TANGKAP DATA DARI ANDROID (GAMBAR & METADATA)
        if 'image' not in request.files:
            return jsonify({"error": "Tidak ada gambar yang dikirim"}), 400
        
        file = request.files['image']
        
        # Tangkap data pasien (Usia, Gender, Lokasi) - dikirim sebagai form-data
        age = float(request.form.get('age', 0))
        sex = int(request.form.get('sex', 0)) # 0: Female, 1: Male
        lokasi = request.form.get('lokasi', 'head/neck')
        
        # 3. PRA-PEMROSESAN CITRA (Sama seperti Colab)
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0 # Normalisasi 0-1
        img_array = np.expand_dims(img_array, axis=0) # Tambah dimensi batch
        
        # 4. PRA-PEMROSESAN METADATA (Sama seperti Colab)
        # Normalisasi usia (Min-Max: min=5, max=85 asumsi)
        age_norm = (age - 5) / (85 - 5)
        age_norm = max(0.0, min(1.0, age_norm)) # Pastikan di rentang 0-1
        
        # One-Hot Encoding Lokasi Lesi (4 Kategori)
        loc_head = 1.0 if lokasi == 'head/neck' else 0.0
        loc_upper = 1.0 if lokasi == 'upper extremity' else 0.0
        loc_lower = 1.0 if lokasi == 'lower extremity' else 0.0
        loc_oral = 1.0 if lokasi == 'oral/genital' else 0.0
        
        # Gabungkan metadata menjadi array 6 dimensi
        meta_array = np.array([[age_norm, sex, loc_head, loc_lower, loc_oral, loc_upper]], dtype=np.float32)
        
        # 5. MASUKKAN KE DALAM MODEL TFLITE
        # Karena model hybrid punya 2 input, kita harus cek index inputnya
        # Asumsi: input_details[0] adalah gambar, input_details[1] adalah klinis (atau sebaliknya)
        for detail in input_details:
            if 'visual' in detail['name'].lower() or len(detail['shape']) == 4:
                interpreter.set_tensor(detail['index'], img_array)
            elif 'klinis' in detail['name'].lower() or len(detail['shape']) == 2:
                interpreter.set_tensor(detail['index'], meta_array)
                
        interpreter.invoke()
        
        # 6. AMBIL HASIL PREDIKSI
        output_data = interpreter.get_tensor(output_details[0]['index'])
        probabilities = output_data[0]
        
        max_index = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[max_index]
        confidence = float(probabilities[max_index]) * 100
        
        # Kirim jawaban format JSON ke Android
        return jsonify({
            "status": "success",
            "diagnosis": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "message": "Diagnosis berhasil dilakukan"
        })
        
    except Exception as e:
        print("\n====== ALARM ERROR AI ======")
        print(str(e))
        print("============================\n")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Jalankan server
    app.run(debug=True, host='0.0.0.0', port=5000)