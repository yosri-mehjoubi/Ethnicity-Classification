import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from PIL import Image
from io import BytesIO
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


# Define ethnicities
ethnicities = ['Caucasian ', 'East-asian ', 'Latino ', 'Middle-Eastern ', 'North-African ', 'South-asian ', 'Subsaharian ']

# Load the trained ethnicity classification model
model = load_model('ethnicity_classification.h5')

# Assuming you have defined these functions and loaded your data appropriately
def preprocess_image(img):
    img = img / 255.0  # Normalize pixel values
    return img

# set page layout
st.set_page_config(
    page_title="Ethnicity Classification App",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title(" Welcome to Advanced AI Deep Learning for Ethnicity Classification")
st.sidebar.subheader("Input")

# component to upload images
uploaded_file = st.sidebar.file_uploader(
    "Choose an image to classify", type=["jpg", "jpeg", "png"]
)
# component for toggling confusion matrix
show_confusion_matrix = st.sidebar.checkbox("Show Confusion Matrix")

if uploaded_file:
    bytes_data = uploaded_file.read()

    # load the input image using PIL image utilities
    image = Image.open(BytesIO(bytes_data))
    image = image.convert("RGB")
    image = image.resize((224, 224))  # Assuming the model is trained on this input size
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    preprocessed_image = preprocess_image(image_array)

    # make prediction
    predicted_class_index = np.argmax(model.predict(preprocessed_image), axis=1)[0]
    predicted_class_label = ethnicities[predicted_class_index]

    st.image(bytes_data, caption=[f"Predicted Class: {predicted_class_label}"])
    # image_path = st.file_uploader("Sélectionnez une image", type=["jpg", "jpeg", "png"])
    if True:
        # img = load_img(image_path, target_size=(224, 224))
        # img_array = img_to_array(img)
        # img_array = np.expand_dims(img_array, axis=0)

        # # Normalisation de l'image
        # img_array /= 255.0

        # Prédiction
        predictions = model.predict(preprocessed_image)

        # Noms des ethnies
        ethnicities = ['Caucasian ', 'East-asian ', 'Latino ', 'Middle-Eastern ', 'North-African ', 'South-asian ', 'Subsaharian ']

        # Afficher les pourcentages prédits pour chaque ethnie
        st.write("Résultats de la prédiction :")
        for i in range(len(ethnicities)):
            st.write(f"{ethnicities[i]}: {predictions[0][i] * 100:.2f}%")

        # Afficher les résultats sous forme de graphique à barres
        st.bar_chart({ethnicities[i]: predictions[0][i] * 100 for i in range(len(ethnicities))})
        # Display confusion matrix if selected
    if show_confusion_matrix:
        # Assuming you have ground truth labels (actual classes)
        actual_label = st.sidebar.selectbox("Select Actual Label", ethnicities)  # Adjust based on your classes

        # Assuming you have a dataset with actual labels and predicted labels
        # Replace this part with your actual data
        actual_labels = [actual_label]
        predicted_labels = [predicted_class_label]

        # Calculate confusion matrix
        cm = confusion_matrix(actual_labels, predicted_labels, labels=ethnicities)

        # Display confusion matrix
        st.subheader("Confusion Matrix")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ethnicities)
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        st.pyplot(fig)
