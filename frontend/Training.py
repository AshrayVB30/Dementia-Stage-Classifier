import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os
import cv2
import tensorflow as tf
from tqdm import tqdm
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing import image

# Constants
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
CLASS_LABELS = {name: i for i, name in enumerate(CLASS_NAMES)}
IMAGE_SIZE = (150, 150)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dementia_model.h5')

def load_data(base_path='datasets'):
    datasets = [os.path.join(base_path, 'Atrain'), os.path.join(base_path, 'Atest')]
    output = []

    for dataset in datasets:
        images = []
        labels = []
        print("Loading", dataset)

        # Loop through 'train' and 'test' subdirectories inside Atrain and Atest
        for subfolder in ['train', 'test']:
            subfolder_path = os.path.join(dataset, subfolder)
            print(f"Processing subfolder: {subfolder_path}")

            for folder in os.listdir(subfolder_path):
                label = CLASS_LABELS.get(folder)
                if label is None:
                    continue

                folder_path = os.path.join(subfolder_path, folder)
                print(f"Processing class: {folder} in {subfolder_path}")

                # Ensure the folder contains images
                if not os.listdir(folder_path):
                    print(f"Warning: No images found in {folder_path}")

                for file in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, file)
                    image_bgr = cv2.imread(img_path)

                    if image_bgr is None:
                        continue  # Skip unreadable images

                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    image_resized = cv2.resize(image_rgb, IMAGE_SIZE)
                    images.append(image_resized)
                    labels.append(label)

        images = np.array(images, dtype='float32') / 255.0
        labels = np.array(labels, dtype='int32')
        print(f"Loaded {len(images)} images.")
        output.append((images, labels))

    return output


def train_model():
    (train_images, train_labels), (test_images, test_labels) = load_data()
    train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, batch_size=128, epochs=10, validation_split=0.2)

    model.save(MODEL_PATH)
    print("Model saved to", MODEL_PATH)


def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        print("⚠️ Model file not found at", MODEL_PATH)
        return None


def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    return CLASS_NAMES[predicted_label], confidence


def get_suggestions(pred_label):
    suggestions_map = {
        'MildDemented': (
            "BEHAVIOR: Increased forgetfulness and confusion",
            "PRECAUTIONS: Build a strong support network of family and friends"
        ),
        'ModerateDemented': (
            "BEHAVIOR: Significant memory loss, including forgetting personal history",
            "PRECAUTIONS: Increased need for caregiving support, possibly professional care"
        ),
        'NonDemented': (
            "BEHAVIOR: Occasional forgetfulness, but not interfering with daily life",
            "PRECAUTIONS: Maintain regular mental activities and healthy lifestyle"
        ),
        'VeryMildDemented': (
            "BEHAVIOR: Early signs of memory lapses",
            "PRECAUTIONS: Start routine cognitive exercises and healthy habits"
        )
    }
    return suggestions_map.get(pred_label, ("Unknown", "No suggestion available."))


# Optional: run training when executed directly
if __name__ == "__main__":
    train_model()
