import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = "dataset"  # Must contain 'train' and 'test' folders
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Parameters
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10  # Increase for better accuracy

# -----------------------------
# Data Loading & Augmentation
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# -----------------------------
# Save class labels
# -----------------------------
labels = list(train_gen.class_indices.keys())
with open("labels.txt", "w") as f:
    for label in labels:
        f.write(f"{label}\n")
print("✅ labels.txt saved:", labels)

# -----------------------------
# Build Model (Transfer Learning)
# -----------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=EPOCHS
)

# -----------------------------
# Evaluate Model
# -----------------------------
loss, acc = model.evaluate(test_gen)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# -----------------------------
# Save & Convert to TFLite
# -----------------------------
saved_model_path = os.path.join(MODEL_DIR, "currency_model.keras")
model.save(saved_model_path)
print(f"✅ Keras model saved to: {saved_model_path}")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_path = os.path.join(MODEL_DIR, "currency_model.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved to: {tflite_path}")
