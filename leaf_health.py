import streamlit as st
from PIL import Image

# import matplotlib.pyplot as plt
# import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras import preprocessing
import time


## this is part of web app

## ----------------------------------------------- x -----------------------------------------x-------------------------x------------------##


# fig = plt.figure()
def add_bg_from_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/premium-vector/abstract-lungs-polygonal-digital-lines-pathogens-concept-lung-disease-tb-lung-cancer_36402-1273.jpg?w=1060");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True,
    )


add_bg_from_url()


st.title("Lung Cancer Detection")

st.markdown("Prediction : (Adenocarcinoma  or  Squamous Cell Carcinoma)")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner("Model working...."):
                # plt.imshow(image)
                # plt.axis("off")

                predictions = predict(image)

                time.sleep(1)
                st.success("Classified")
                st.write(predictions)


## This code is for saved model in format as H5 file


def predict(image):
    classifier_model = "leafdisease_CNN.h5"

    model = load_model(classifier_model)

    test_image = image.resize((256, 256))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = {0: "Adenocarcinoma", 1: "Squamous Cell Carcinoma"}

    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()

    result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence."
    return result


## -----------------------------------------------------x---------------------------------------x--------------------------------------------##


## this code for format tflite file
# def predict(image):
#     model = "leaves_model.tflite"


#     interpreter = tf.lite.Interpreter(model_path = model)
#     interpreter.allocate_tensors()
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     input_shape = input_details[0]['shape']
#     image = np.array(image.resize((200,200)), dtype=np.float32)

#     image = image / 255.0
#     image = np.expand_dims(image, axis=0)

#     interpreter.set_tensor(input_details[0]['index'], image)
#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     probabilities = np.array(output_data[0])

#     labels = {0 : "healthy", 1 : "diseased"}

#     label_to_probabilities = []

#     for i, probability in enumerate(probabilities):
#         label_to_probabilities.append([labels[i], float(probability)])

#     sorted(label_to_probabilities, key=lambda element: element[1])

#     result = { 'healthy' : 0 , 'diseased' : 0 }

#     result = f"{label_to_probabilities[np.argmax(probability)][0]} with a { (100 * np.max(probabilities)).round(2)} % confidence."

#     return result


if __name__ == "__main__":
    main()
