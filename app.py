import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="wide"
)

# Load trained models
@st.cache_resource
def load_models():
    alphabet_model = load_model("handwritten_alphabet_model.keras")
    digit_model = load_model("handwritten_digit_model.keras")
    return alphabet_model, digit_model

alphabet_model, digit_model = load_models()

# Title
st.title("✍️ Handwritten Character Recognition")
st.write("Recognize handwritten alphabets and digits using CNN.")

st.divider()

# Select recognition type
choice = st.selectbox(
    "Choose Recognition Type",
    ["Alphabet Recognition (A-Z)", "Digit Recognition (0-9)"]
)

# Upload image
upload_tab, camera_tab = st.tabs(["📁 Upload Image", "📷 Take Photo"])

with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload a handwritten image",
        type=["png", "jpg", "jpeg"],
        key="upload"
    )

with camera_tab:
    camera_file = st.camera_input(
        "Capture a handwritten alphabet or digit",
        key="camera"
    )

if camera_file is not None:
    uploaded_file = camera_file

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.subheader("Uploaded Image")
    st.image(image, width=250)

    if st.button("Predict"):

        # Preprocess image
        img_array = np.array(image)

        if img_array.mean() > 127:
            img_array = 255 - img_array

        rows, cols = np.where(img_array > 50)

        if len(rows) == 0:
            st.error("No handwritten character found.")

        else:

            img_array = img_array[
                rows.min():rows.max() + 1,
                cols.min():cols.max() + 1
            ]

            letter_img = Image.fromarray(
                img_array.astype("uint8")
            )

            letter_img.thumbnail((20, 20))

            canvas = Image.new("L", (28, 28), 0)

            x = (28 - letter_img.width) // 2
            y = (28 - letter_img.height) // 2

            canvas.paste(letter_img, (x, y))

            img_array = np.array(
                canvas,
                dtype="float32"
            ) / 255.0

            img_array = img_array.reshape(1, 28, 28, 1)

            # Get predictions from BOTH models
            alphabet_prediction = alphabet_model.predict(
                img_array,
                verbose=0
            )[0]

            digit_prediction = digit_model.predict(
                img_array,
                verbose=0
            )[0]

            # Find highest confidence from each model
            alphabet_confidence = np.max(alphabet_prediction) * 100
            digit_confidence = np.max(digit_prediction) * 100

            # Check selected recognition type
            if choice == "Alphabet Recognition (A-Z)":

                # Check whether the image looks more like a digit
                if digit_confidence > alphabet_confidence + 15 and digit_confidence >= 60:

                    st.error(
                        "⚠️ Please upload an alphabet image (A-Z). "
                        "The uploaded image appears to be a digit."
                    )

                else:

                    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                    prediction = alphabet_prediction

                    predicted_index = np.argmax(prediction)
                    predicted_character = labels[predicted_index]

                    confidence = prediction[predicted_index] * 100

                    st.success("Prediction Completed!")

                    st.subheader(
                        f"Predicted Alphabet: {predicted_character}"
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                    # Top 5 predictions
                    st.subheader("Top 5 Predictions")

                    top_indices = np.argsort(
                        prediction
                    )[-5:][::-1]

                    for index in top_indices:
                        st.write(
                            f"{labels[index]} : "
                            f"{prediction[index] * 100:.2f}%"
                        )

                    st.image(
                        canvas,
                        caption="Processed Image",
                        width=150
                    )

            else:

                # Check whether the image looks more like an alphabet
                if alphabet_confidence > digit_confidence + 15 and alphabet_confidence >= 60:

                    st.error(
                        "⚠️ Please upload a digit image (0-9). "
                        "The uploaded image appears to be an alphabet."
                    )

                else:

                    labels = list("0123456789")
                    prediction = digit_prediction

                    predicted_index = np.argmax(prediction)
                    predicted_character = labels[predicted_index]

                    confidence = prediction[predicted_index] * 100

                    st.success("Prediction Completed!")

                    st.subheader(
                        f"Predicted Digit: {predicted_character}"
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                    # Top 5 predictions
                    st.subheader("Top 5 Predictions")

                    top_indices = np.argsort(
                        prediction
                    )[-5:][::-1]

                    for index in top_indices:
                        st.write(
                            f"{labels[index]} : "
                            f"{prediction[index] * 100:.2f}%"
                        )

                    st.image(
                        canvas,
                        caption="Processed Image",
                        width=150
                    )

st.divider()

st.caption(
    "Developed using Python, TensorFlow, CNN and Streamlit."
)