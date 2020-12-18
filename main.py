import streamlit as st
from ResNet_EfficientNet_House_clf import *
from Semantic_Segmentation_code import Segmentator

import numpy as np
import pandas as pd

from PIL import Image


st.title('House`s Images Processor')

st.markdown('''**Image Classifier:**
Classification by 4 next classes:
 - Cosmetic Repair
 - Luxury
 - Standart
 - without modify''')


uploaded_file_clf = st.file_uploader("Choose a file to Classification model", key="1")
if uploaded_file_clf is not None:
    image = Image.open(uploaded_file_clf)
    st.image(image, caption="started image", width=700)

    clf = ResNetAndEfficientNetClf()

    image_2, predictions = clf.get_predictions(clf.model, image)
    pred = predictions[0].argmax()
    st.write("Prediction: " + clf.class_names[pred])


st.markdown('''**Semantic Segmentation:**
Getting the location of objects in the image''')

uploaded_file_segmet = st.file_uploader("Choose a file to Classification model", key="2")
if uploaded_file_segmet is not None:
    image_2 = Image.open(uploaded_file_segmet)
    st.image(image_2, caption="started image", width=700)

    segment = Segmentator()

    image_22 = segment.main_get_results(image_2)
    st.image(image_22, caption="processed image", width=700)