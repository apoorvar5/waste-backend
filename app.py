from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import tempfile
from flask_cors import CORS



# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Use explicit path to the model directory
#model_path = r"C:\Users\ACER\Desktop\Waste Classification\trash_classifier_mobilenetv2"

model_path = os.path.join(os.path.dirname(__file__), "trash_classifier_mobilenetv2")


print(f"Looking for model at: {model_path}")
if os.path.exists(model_path):
    print(f"Path exists. Files in directory: {os.listdir(model_path)}")
else:
    print("Path does not exist!")

# Load the trained model
try:
    model = tf.saved_model.load(model_path)
    infer = model.signatures['serving_default']
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# Define the class labels
class_labels = ['Food Waste', 'Landfill', 'Recycle']

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image is part of the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    # If no file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        # Process the image directly from memory without saving
        img = Image.open(file.stream)
        img = img.resize((224, 224))  # Resize image to match model input
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Get model prediction using the SavedModel signature
        input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        predictions = infer(input_tensor)
        
        # Debug output
        print(f"Prediction output keys: {list(predictions.keys())}")
        
        # Get the output tensor (the name might vary)
        output_key = list(predictions.keys())[0]
        output_tensor = predictions[output_key].numpy()
        
        predicted_class = np.argmax(output_tensor, axis=1)[0]
        confidence = float(output_tensor[0][predicted_class])
        
        # Return the prediction as JSON
        return jsonify({
            'prediction': class_labels[predicted_class],
            'confidence': confidence,
            'class_index': int(predicted_class)
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in prediction: {error_trace}")
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)