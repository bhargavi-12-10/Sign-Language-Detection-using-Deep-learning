import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import math
from gtts import gTTS

# Load the pre-trained model
MODEL_PATH = 'sign_language_model.h5'
model = load_model(MODEL_PATH)

# Image parameters
img_height, img_width = 128, 128
imgSize = 300
offset = 20

# Function for text-to-speech
def text_to_speech(text, output_file):
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)

# Function to make predictions
def predict_class(img_file):
    img = image.load_img(img_file, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    class_label = class_names[class_idx]  # Get class label based on index
    return class_label

# Extract class names from the model training
class_names = ['Hello', 'I Love You', 'No', 'Stop', 'Thank You', 'Thumbs Down', 'Thumbs Up', 'Victory', 'Yes']

# Function to crop hand and save image
def process_and_crop_hand(image_file, save_path="image_snap.png"):
    detector = HandDetector(maxHands=1)
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        if 0 <= y - offset and y + h + offset <= img.shape[0] and \
           0 <= x - offset and x + w + offset <= img.shape[1]:
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize
            
            # Save the processed image
            cv2.imwrite(save_path, imgWhite)
            return save_path
    return None

# Streamlit UI
st.title("Sign Language Classification")
st.write("Upload an image or take a photo to predict the corresponding sign language class.")

# Option Selection
option = st.radio("Select Input Method:", ("Upload Image", "Take Photo"))

if option == "Upload Image":
    # File uploader
    uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
    if uploaded_image is not None:
        st.image(uploaded_image, caption='Uploaded Image', use_container_width=True)
        st.write("Processing...")
        
        # Process and crop the hand
        processed_image_path = process_and_crop_hand(uploaded_image, "temp_image.png")
        if processed_image_path:
            prediction = predict_class(processed_image_path)
            st.write(f"Predicted Class: *{prediction}*")
            
            # Generate and play audio for the prediction
            audio_file = "voice_response.mp3"
            text_to_speech(prediction, audio_file)
            st.audio(audio_file, format="audio/mp3")
        else:
            st.write("No hand detected. Please try again.")

elif option == "Take Photo":
    # Camera input
    camera_image = st.camera_input("Take a photo")
    if camera_image is not None:
        st.image(camera_image, caption='Captured Image', use_container_width=True)
        st.write("Processing...")
        
        # Process and crop the hand
        processed_image_path = process_and_crop_hand(camera_image, "temp_shot.png")
        if processed_image_path:
            prediction = predict_class(processed_image_path)
            st.write(f"Predicted Class: *{prediction}*")
            
            # Generate and play audio for the prediction
            audio_file = "voice_response.mp3"
            text_to_speech(prediction, audio_file)
            st.audio(audio_file, format="audio/mp3")
        else:
            st.write("No hand detected. Please try again.")