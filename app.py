import streamlit as st
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import os
import time, datetime
from PIL import Image

FACE_DB = "face_db"
os.makedirs(FACE_DB, exist_ok=True)

def add_face(image, name):
    person_dir = os.path.join(FACE_DB, name)
    os.makedirs(person_dir, exist_ok=True)
    image_path = os.path.join(person_dir, f"{name}_{int(time.time())}.jpg")
    cv2.imwrite(image_path, image)



# Streamlit UI
st.title("Sign Language Recognition")

menu = st.sidebar.selectbox("Choose Module", ["Add Face", "Sign Language Detection"])

if menu == "Add Face":
    st.header("Add Face to Database")
    option = st.selectbox("Choose Option", ["Upload Image", "Take Snapshot"])

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None and st.button("Add Face"):
            image = Image.open(uploaded_file)
            image = np.array(image)
            add_face(image, name)

    elif option == "Take Snapshot":
        snapshot = st.camera_input("Capture Image")

        if snapshot is not None and st.button("Add Face"):
            image = Image.open(snapshot)
            image = np.array(image)
            add_face(image, name)
            
elif menu == "Sign Language Detection":
    st.header("Sign Language Detection")
