import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model

# Load trained alphabet recognition model
model = load_model("handwritten_alphabet_model.keras")

# Letters in EMNIST Letters dataset
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Input image
image_path = "my_letter.png"

# Open image and convert to grayscale
img = Image.open(image_path).convert("L")

# Convert image to NumPy array
img_array = np.array(img)

# Make background black and letter white
if img_array.mean() > 127:
    img_array = 255 - img_array

# Find letter boundaries
rows, cols = np.where(img_array > 50)

if len(rows) == 0:
    raise ValueError("No letter found! Check your image.")

# Crop the letter
img_array = img_array[
    rows.min():rows.max() + 1,
    cols.min():cols.max() + 1
]

# Resize while maintaining proportions
letter_img = Image.fromarray(img_array.astype("uint8"))
letter_img.thumbnail((20, 20))

# Create 28x28 black canvas
canvas = Image.new("L", (28, 28), 0)

x = (28 - letter_img.width) // 2
y = (28 - letter_img.height) // 2

canvas.paste(letter_img, (x, y))

# Normalize image
img_array = np.array(canvas, dtype="float32") / 255.0

# Prepare image for prediction
img_array = img_array.reshape(1, 28, 28, 1)

# Predict the letter
prediction = model.predict(img_array, verbose=0)

# Display top 5 predictions
top_indices = np.argsort(prediction[0])[-5:][::-1]

print("\nTop 5 predictions:")

for index in top_indices:
    print(f"{letters[index]}: {prediction[0][index] * 100:.2f}%")

# Get predicted letter
predicted_index = np.argmax(prediction)

print("\nPredicted letter:", letters[predicted_index])

# Display processed image
plt.imshow(canvas, cmap="gray")
plt.title(f"Predicted letter: {letters[predicted_index]}")
plt.axis("off")
plt.show()