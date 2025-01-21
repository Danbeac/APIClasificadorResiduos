# filepath: /C:/Users/danbe/Downloads/DevWork/Proyecto_Clasificacion_Residuos/Modelos/IA_V1_X/API_IA_MODEL.py
import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import base64
from flask import Flask, request, jsonify
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Habilitar CORS para todas las rutas

# Cargar el modelo
model = load_model("C:/Users/danbe/Downloads/DevWork/Proyectos Github\APIClasificadorResiduosPY/keras_model.h5", compile=False)

# Cargar las etiquetas
class_names = open("C:/Users/danbe/Downloads/DevWork/Proyectos Github\APIClasificadorResiduosPY/labels.txt", "r").readlines()

def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener la imagen en base64 del request
        image_b64 = request.json['image']
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))

        # Preprocesar la imagen
        processed_image = preprocess_image(image)

        # Realizar la predicción
        prediction = model.predict(processed_image)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        # Retornar el resultado
        return jsonify({
            'class_name': class_name,
            'confidence_score': float(confidence_score)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)