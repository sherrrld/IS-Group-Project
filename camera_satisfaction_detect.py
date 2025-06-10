import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model and emotion labels
model = load_model("best_model.keras")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Define satisfaction categories
satisfied_emotions = ['happy', 'neutral', 'surprise']
unsatisfied_emotions = ['angry', 'disgust', 'fear', 'sad']

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = np.reshape(face_normalized, (1, 48, 48, 1))

        prediction = model.predict(face_input, verbose=0)
        predicted_emotion = emotion_labels[np.argmax(prediction)]

        # Map to final output
        if predicted_emotion in satisfied_emotions:
            label = "Prediction: Satisfied"
            color = (0, 255, 0)
        else:
            label = "Prediction: Unsatisfied"
            color = (0, 0, 255)

        # Draw results
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Satisfaction Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
