import streamlit as st
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Define a dictionary to map class indices to custom class labels
class_labels = {
    0: 'antelope', 1: 'badger', 2: 'bat', 3: 'bear', 4: 'bee', 5: 'beetle', 6: 'bison', 7: 'boar',
    8: 'butterfly', 9: 'cat', 10: 'caterpillar', 11: 'chimpanzee', 12: 'cockroach', 13: 'cow',
    14: 'coyote', 15: 'crab', 16: 'crow', 17: 'deer', 18: 'dog', 19: 'dolphin', 20: 'donkey',
    21: 'dragonfly', 22: 'duck', 23: 'eagle', 24: 'elephant', 25: 'flamingo', 26: 'fly',
    27: 'fox', 28: 'goat', 29: 'goldfish', 30: 'goose', 31: 'gorilla', 32: 'grasshopper',
    33: 'hamster', 34: 'hare', 35: 'hedgehog', 36: 'hippopotamus', 37: 'hornbill', 38: 'horse',
    39: 'hummingbird', 40: 'hyena', 41: 'jellyfish', 42: 'kangaroo', 43: 'koala', 44: 'ladybugs',
    45: 'leopard', 46: 'lion', 47: 'lizard', 48: 'lobster', 49: 'mosquito', 50: 'moth', 51: 'mouse',
    52: 'octopus', 53: 'okapi', 54: 'orangutan', 55: 'otter', 56: 'owl', 57: 'ox', 58: 'oyster',
    59: 'panda', 60: 'parrot', 61: 'pelecaniformes', 62: 'penguin', 63: 'pig', 64: 'pigeon',
    65: 'porcupine', 66: 'possum', 67: 'raccoon', 68: 'rat', 69: 'reindeer', 70: 'rhinoceros',
    71: 'sandpiper', 72: 'seahorse', 73: 'seal', 74: 'shark', 75: 'sheep', 76: 'snake', 77: 'sparrow',
    78: 'squid', 79: 'squirrel', 80: 'starfish', 81: 'swan', 82: 'tiger', 83: 'turkey', 84: 'turtle',
    85: 'whale', 86: 'wolf', 87: 'wombat', 88: 'woodpecker', 89: 'zebra'
}

# Specify the path to your custom model
model_path = r"C:\Users\sandeep\OneDrive\Desktop\Navya_files\Mini Project\mobilenet_v2.h5"  # Replace with the actual path to your model file

# Load the custom model
model = MobileNetV2(weights=None)
model.load_weights(model_path)

st.title("Image Classification on Animals")
st.write("Upload an image for classification")

# Upload image through Streamlit
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Preprocess the image
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Make a prediction
    preds = model.predict(img)
    decoded_preds = decode_predictions(preds, top=1)[0]

    # Extract the class label and confidence
    predicted_class_label = decoded_preds[0][1]
    confidence = decoded_preds[0][2]

    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    st.write(f"Prediction: {predicted_class_label}")
    st.write(f"Accuracy: {confidence:.2f}")
