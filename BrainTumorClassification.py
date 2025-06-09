import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import networkx as nx
import requests

#Directory
train_dir = "Training"
test_dir = "Testing"
img_height, img_width = 299, 299  #Xception recommended input size
batch_size = 32

#Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2,
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

#Train, Validation, and Test Generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)

#Xception base model
base_model = Xception(
    weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3), pooling="max"
)

#Build model
model = Sequential(
    [
        Input(shape=(img_height, img_width, 3)),  #Explicit input layer
        base_model,
        Flatten(),
        Dropout(rate=0.3),
        Dense(128, activation="relu"),
        Dropout(rate=0.25),
        Dense(4, activation="softmax"),  #4 classes
    ]
)

#Compile model
model.compile(
    optimizer=Adamax(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
)

#Callback (cuts down time by decent bit)
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-5)

#Train model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr],
)

#Plot Training and Validation Accuracy
history_dict = history.history

plt.figure(figsize=(10, 5))
plt.plot(history_dict["accuracy"], label="Training Accuracy")
plt.plot(history_dict["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid()
plt.show()

#Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(history_dict["loss"], label="Training Loss")
plt.plot(history_dict["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.show()

#Evaluate model on test set
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

#Predictions on test set for confusion matrix and classification report
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

#Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
labels = list(test_generator.class_indices.keys())
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

#Classification Report
clr = classification_report(y_true, y_pred_classes, target_names=labels)
print(clr)

#Knowledge Graph Integration
static_knowledge_graph = {
    "Glioma": ["Headache", "Seizures"],
    "Meningioma": ["Headache"],
    "Pituitary Tumor": ["Vision Loss", "Hormonal Imbalance"],
    "No Tumor": ["None"]
}

#Fetch Symptoms from Google Knowledge Graph API
def fetch_symptoms_from_google(tumor_type, api_key):
    query = f"{tumor_type} tumor symptoms"
    url = "https://kgsearch.googleapis.com/v1/entities:search"
    params = {
        "query": query,
        "key": api_key,
        "limit": 10,
        "types": "Thing",
    }
    response = requests.get(url, params=params)
    data = response.json()
    print(f"Debug: Google API Response for '{tumor_type}':\n{data}\n")
    if "itemListElement" in data and len(data["itemListElement"]) > 0:
        result = data["itemListElement"][0].get("result", {})
        description = result.get("detailedDescription", {}).get("articleBody", "No details found.")
        return description
    print(f"No API results for {tumor_type}, falling back to static knowledge graph.")
    return ", ".join(static_knowledge_graph.get(tumor_type.title(), ["No symptoms available."]))

#Query Knowledge Graph for Symptoms
def get_related_symptoms(tumor_type):
    tumor_type = tumor_type.title()
    if tumor_type in static_knowledge_graph:
        return ", ".join(static_knowledge_graph[tumor_type])
    return "No related symptoms found."

#Diagnose Tumor Type and Fetch Symptoms
def diagnose_with_knowledge(image_path, model, class_map, api_key):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    tumor_type = class_map[predicted_class]
    symptoms_info = fetch_symptoms_from_google(tumor_type, api_key)
    return tumor_type, symptoms_info

#Example
class_map = {v: k for k, v in test_generator.class_indices.items()}
image_path = "/Users/kevinou/Desktop/Grad/Projects/BrainTumorClassification/Testing/glioma/Te-gl_0010.jpg"
api_key = ""
tumor_type, related_symptoms = diagnose_with_knowledge(image_path, model, class_map, api_key)
print(f"Predicted Tumor Type: {tumor_type}")
print(f"Related Symptoms: {related_symptoms}")
