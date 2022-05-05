import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np


@st.cache(allow_output_mutation=True)

def load_model():
    model=tf.keras.models.load_model("C:\\Users\\Abhishek\\Desktop\\Malaria-Infected-Cell-Classification-main\\models\\mymodel.hdf5")
    return model
with st.spinner('LOADING..'):
    model=load_model()
    
st.write("""
         MALARIA DETECTION
         """
         )

file = st.file_uploader("UPLOAD IMAGE", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (68,68)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    
    return prediction


    
if file is None:
    st.text("UPLOAD IMAGE")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
  
    if predictions == [[1.]]:
        string = "HEALTHY"
    else:
        string = "MALARIA DETECTED"
        
    st.success(string)
    
        
    