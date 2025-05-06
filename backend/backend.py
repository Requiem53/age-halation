from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from io import BytesIO
import requests
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the processor and model
processor = AutoImageProcessor.from_pretrained("processor_directory")  # Local saved processor
model = AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier")
model.load_state_dict(torch.load("vit_age_classifier.pkl", map_location=torch.device("cpu")))
model.eval()

@app.route('/keepalive', methods=['GET'])
def api_health():
    return jsonify(Message="Success")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Support Option A: JSON with image_url
        if request.is_json:
            data = request.get_json()
            image_url = data.get("image_url")
            if not image_url:
                return jsonify({"error": "Missing 'image_url' in JSON"}), 400
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")

        # Support Option B: form-data with image file
        elif 'image' in request.files:
            image_file = request.files['image']
            image = Image.open(image_file).convert("RGB")

        else:
            return jsonify({"error": "No image provided"}), 400

        # Preprocess and predict
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_label = logits.argmax(-1).item()
        label_name = model.config.id2label[predicted_label]

        return jsonify({
            "predicted_class_id": predicted_label,
            "predicted_label": label_name
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run app
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
