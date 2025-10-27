import cv2
import numpy as np
import tensorflow as tf
import pyttsx3

# -----------------------------
# Load model and labels
# -----------------------------
MODEL_PATH = "models/currency_model.tflite"
LABELS_PATH = "labels.txt"

# Load class labels
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape'][1:3]  # (224, 224)

# -----------------------------
# Initialize TTS
# -----------------------------
engine = pyttsx3.init()
engine.setProperty("rate", 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# -----------------------------
# Live camera feed
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("✅ Starting camera... show a currency note in front of it.")
speak("Camera started. Please show a currency note.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Preprocess frame for model
    img = cv2.resize(frame, input_shape)
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    confidence = np.max(output_data)

    label = labels[predicted_class]
    text = f"{label} ({confidence*100:.1f}%)"

    # Display result
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Currency Detection", frame)

    # Speak result if confidence > 80%
    if confidence > 0.80:
        speak(f"This is a {label} rupees note")
        print(f"Detected: {label} ({confidence*100:.1f}%)")

    # Exit when 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
speak("Exiting live detection. Goodbye!")
