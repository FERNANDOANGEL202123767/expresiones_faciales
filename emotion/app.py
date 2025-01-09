import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Configurar backend antes de importar pyplot
import matplotlib.pyplot as plt
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from io import BytesIO
import base64
from pyngrok import ngrok

# Crear la aplicación Flask
app = Flask(__name__)

# Configurar carpeta de subida
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB de tamaño máximo de archivo

# Asegurar que exista la carpeta de subida
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Verificar si el archivo tiene una extensión permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def string2array(x):
    return np.array(x.split(' ')).reshape(48, 48, 1).astype('float32')

def resize(x):
    img = x.reshape(48, 48)
    return cv2.resize(img, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)

def load_dataset():
    facialexpression_df = pd.read_csv('icml_face_data.csv')

    # Convertir píxeles de string a array
    facialexpression_df[' pixels'] = facialexpression_df[' pixels'].apply(lambda x: string2array(x))

    # Redimensionar imágenes
    facialexpression_df[' pixels'] = facialexpression_df[' pixels'].apply(lambda x: resize(x))
    return facialexpression_df

@app.route('/explore', methods=['GET'])
def explore_dataset():
    try:
        # Cargar el dataset
        df = load_dataset()

        # Verificar estructura del dataset
        data_info = {
            'shape': df.shape,
            'null_values': df.isnull().sum().to_dict(),
            'emotion_counts': df['emotion'].value_counts().to_dict()
        }

        # Visualizar una muestra del dataset
        sample_image = df[' pixels'][0]
        plt.figure()
        plt.imshow(sample_image, cmap='gray')
        plt.title('Ejemplo de Imagen')

        # Guardar figura en base64
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        return jsonify({
            'success': True,
            'data_info': data_info,
            'sample_image': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/plot_emotions', methods=['GET'])
def plot_emotions():
    try:
        df = load_dataset()

        # Graficar distribución de emociones
        plt.figure(figsize=(10, 6))
        sns.barplot(x=df['emotion'].value_counts().index, y=df['emotion'].value_counts())
        plt.title('Distribución de Emociones')
        plt.xlabel('Emoción')
        plt.ylabel('Número de muestras')

        # Guardar figura en base64
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        return jsonify({
            'success': True,
            'emotion_plot': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    images = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            images.append(filename)
    return render_template('index.html', images=images)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Ruta para analizar imágenes con una operación específica."""
    try:
        # Obtener la operación seleccionada
        operation = request.form.get('operation')
        if operation not in ["original", "flip", "brightness", "flip_vertical"]:
            return jsonify({'error': 'Operación no válida'}), 400

        # Verificar si es un archivo existente o nuevo
        if 'existing_file' in request.form:
            filename = request.form['existing_file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath):
                return jsonify({'error': f'Archivo no encontrado: {filename}'}), 404
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
            if not allowed_file(file.filename):
                return jsonify({'error': 'Tipo de archivo no permitido'}), 400
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        else:
            return jsonify({'error': 'No se proporcionó ningún archivo'}), 400

        # Procesar la imagen
        result_image = process_image(filepath, operation)
        return jsonify({'success': True, 'image': result_image})

    except Exception as e:
        print(f"Error en /analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Ruta para servir archivos subidos."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    try:
        # Iniciar ngrok y Flask
        ngrok_tunnel = ngrok.connect(5001)
        public_url = ngrok_tunnel.public_url
        print(f" * ngrok URL: {public_url}")
        app.run(port=5001)
    except Exception as e:
        print(f"Error al iniciar ngrok o Flask: {e}")