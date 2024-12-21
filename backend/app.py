from flask import Flask, request, jsonify, send_from_directory, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

classification_model = load_model('./model/plant_classifier_model.keras')
freshness_model = load_model('./model/crop_freshness_model.keras')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def predict_freshness(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0) / 255.0 
    
    prediction = model.predict(img_array)[0][0]  
    
    if prediction >= 0.5:
        return "rotten"
    else:
        return "fresh"

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        classification_predictions = classification_model.predict(img_array)
        predicted_class = np.argmax(classification_predictions, axis=-1)[0]

        class_labels = ['aloevera', 'banana', 'bilimbi', 'cantaloupe', 'cassava', 'coconut', 'corn', 'cucumber', 'curcuma', 'eggplant', 'galangal', 'ginger', 'guava', 'kale', 'longbeans', 'mango', 'melon', 'orange', 'paddy', 'papaya', 'peper chili', 'pineapple', 'pomelo', 'shallot', 'soybeans', 'spinach', 'sweet potatoes', 'tobacco', 'waterapple', 'watermelon']  
        predicted_label = class_labels[predicted_class]

        freshness_label = predict_freshness(filepath, freshness_model)

        image_url = url_for('uploaded_file', filename=file.filename)

        return jsonify({
            'classification_prediction': predicted_label,
            'freshness_prediction': freshness_label,
            'image_url': image_url
        }), 200

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('../frontend', filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
