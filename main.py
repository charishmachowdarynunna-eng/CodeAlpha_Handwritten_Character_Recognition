import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Step 1: Load the dataset
print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Step 2: Preprocess the images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("Dataset loaded and preprocessed successfully!")

# Step 3: Build the CNN model
model = Sequential([
    tf.keras.Input(shape=(28, 28, 1)),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(10, activation="softmax")
])

# Step 4: Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN model created successfully!")

# Step 5: Train the model
print("\nTraining started...")

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# Step 6: Evaluate the model
print("\nTesting the model...")

loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Step 7: Predict a digit
index = 0

prediction = model.predict(
    x_test[index:index + 1],
    verbose=0
)

predicted_digit = np.argmax(prediction)

print("Predicted Digit:", predicted_digit)
print("Actual Digit:", y_test[index])

plt.imshow(x_test[index].squeeze(), cmap="gray")
plt.title(f"Predicted: {predicted_digit}")
plt.axis("off")
plt.show()

# Step 8: Save the trained model
model.save("handwritten_digit_model.keras")

print("\nModel saved successfully!")