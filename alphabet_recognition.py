import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the EMNIST Letters dataset
print("Loading EMNIST Letters dataset...")

dataset, info = tfds.load(
    "emnist/letters",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

train_data, test_data = dataset
# Check how an EMNIST image is displayed
for image, label in train_data.take(1):
    plt.imshow(image.numpy().squeeze(), cmap="gray")
    plt.title(f"Label: {label.numpy()}")
    plt.axis("off")
    plt.show()

print("Dataset loaded successfully!")
print("Training images:", info.splits["train"].num_examples)
print("Testing images:", info.splits["test"].num_examples)


# Step 2: Preprocess the images
def preprocess(image, label):
    image = tf.cast(image, tf.float32)

    # Correct the EMNIST image orientation
    image = tf.transpose(image, perm=[1, 0, 2])

    # Normalize pixel values
    image = image / 255.0

    # Ensure the required image shape
    image = tf.ensure_shape(image, (28, 28, 1))

    # Convert labels from 1-26 to 0-25
    label = tf.cast(label, tf.int64) - 1

    return image, label


train_data = train_data.map(preprocess)
test_data = test_data.map(preprocess)

# Shuffle, batch, and prefetch the data
BATCH_SIZE = 128

train_data = (
    train_data
    .shuffle(10000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_data = (
    test_data
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# Step 3: Build the CNN model
print("Building CNN model...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(26, activation="softmax")
])


# Step 4: Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# Step 5: Train the model
print("Training the alphabet recognition model...")

history = model.fit(
    train_data,
    epochs=5,
    validation_data=test_data
)


# Step 6: Evaluate the model
print("Evaluating the model...")

test_loss, test_accuracy = model.evaluate(test_data)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# Step 7: Save the trained model
model.save("handwritten_alphabet_model.keras")

print("Model saved successfully!")


# Step 8: Display an example prediction
for images, labels in test_data.take(1):

    image = images[0:1]

    prediction = model.predict(image, verbose=0)

    predicted_label = int(np.argmax(prediction[0]))
    actual_label = int(labels[0].numpy())

    # Convert labels 0–25 into letters A–Z
    predicted_letter = chr(predicted_label + ord("A"))
    actual_letter = chr(actual_label + ord("A"))

    plt.imshow(image[0].numpy().squeeze(), cmap="gray")
    plt.title(
        f"Predicted: {predicted_letter} | Actual: {actual_letter}"
    )
    plt.axis("off")
    plt.show()

    print("Predicted letter:", predicted_letter)
    print("Actual letter:", actual_letter)
    # Load the saved alphabet recognition model
from tensorflow.keras.models import load_model

model = load_model("handwritten_alphabet_model.keras")

print("Alphabet recognition model loaded successfully!")
