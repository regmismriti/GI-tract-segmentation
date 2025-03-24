import tensorflow as tf
from flask import Flask, request, jsonify
import gradio as gr

import numpy as np
import cv2
from PIL import Image
import io
import os

app = Flask(__name__)

# Load the trained model

# Load the trained segmentation model
MODEL_PATH = "C:/Users/Dell/Desktop/medical image segmentation app/model/resunet.h5"
# Load without compilation
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Recompile with a compatible optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss="binary_crossentropy",  
              metrics=["accuracy"])


# Constants
THRESHOLD = 0.1
IMAGE_SIZE = (320, 320)
LABELS = ["Large Bowel", "Small Bowel", "Stomach"]
class2hexcolor = {"Large Bowel": "#FFFF00" , # yellow
                  "Small Bowel": "#800080", # purple
        
                  "Stomach": "#FF0000"} #red

# Function to preprocess images
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict segmentation masks
def predict_mask(image):
    original_size = image.size
    input_image = preprocess_image(image)
    prediction = model.predict(input_image)[0]  # Remove batch dimension

    masks = []
    for i in range(3):  # Three classes
        mask = (prediction[:, :, i] > THRESHOLD).astype(np.uint8)
        mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        masks.append((mask, LABELS[i]))
    
    return image, masks

# Define Gradio interface
def gradio_interface(image):
    image, masks = predict_mask(image)
    return image, [(mask, label) for mask, label in masks]

with gr.Blocks(title="OncoSegAi") as gradio_app:
    gr.Markdown("""<h1><center>Medical Image Segmentation with ResUNet</center></h1>""")

    with gr.Row():
        img_input = gr.Image(type="pil", label="Input Image")
        img_output = gr.AnnotatedImage(label="Predictions", color_map=class2hexcolor)

    predict_btn = gr.Button("Generate Predictions")
    predict_btn.click(gradio_interface, inputs=img_input, outputs=img_output)

# Flask route to serve Gradio
@app.route("/")
def index():
    return gradio_app.launch(share=True, show_api=False)


# Disable Gradio branding
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

if __name__ == "__main__":
    # gradio_app.launch(share=False, show_api=False)  # Hide Gradio branding
    app.run(host="0.0.0.0", port=5000, debug=True)
