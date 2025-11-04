from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import io
import firebase_admin
from firebase_admin import credentials, db
from model import resnet  # model ArcFace/ResNet của bạn

# ==========================
# 🔧 Firebase setup
# ==========================
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://smartlock-6fb00-default-rtdb.firebaseio.com/"
})

# ==========================
# ⚙️ Flask app setup
# ==========================
app = Flask(__name__)

# ==========================
# 🧠 Load model
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet.to(device)
model.eval()

# ==========================
# 🔍 Helper functions
# ==========================
def cosine_similarity(a, b):
    """Tính độ tương đồng cosine giữa 2 vector"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def preprocess_image(image_bytes):
    """Chuyển bytes → Tensor"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((112, 112)),   # tùy theo input model của bạn
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

# ==========================
# 🔐 Optional: API Key
# ==========================
API_KEY = "esp32_secret_123"

@app.before_request
def verify_api_key():
    key = request.headers.get("x-api-key")
    if key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

# ==========================
# 📸 Endpoint nhận ảnh
# ==========================
@app.route("/verify", methods=["POST"])
def verify():
    try:
        # 1️⃣ Lấy ảnh từ ESP32
        image_bytes = request.data
        img_tensor = preprocess_image(image_bytes)

        # 2️⃣ Trích xuất embedding
        with torch.no_grad():
            emb = model(img_tensor)
            emb = F.normalize(emb).cpu().numpy().flatten()

        # 3️⃣ Lấy embedding từ Firebase
        ref = db.reference("embeddings")  # cấu trúc: embeddings/{name: [vector]}
        data = ref.get()
        if not data:
            return jsonify({"error": "No registered users"}), 404

        # 4️⃣ So sánh embedding
        best_name = None
        best_score = -1
        for name, saved_emb in data.items():
            sim = cosine_similarity(emb, np.array(saved_emb))
            if sim > best_score:
                best_score = sim
                best_name = name

        # 5️⃣ Kiểm tra ngưỡng
        authorized = best_score >= 0.75

        result = {
            "authorized": authorized,
            "name": best_name if authorized else "unknown",
            "score": float(best_score)
        }
        print("✅ Result:", result)
        return jsonify(result)

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

# ==========================
# 🚀 Run Flask
# ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
