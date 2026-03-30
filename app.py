import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load model and class names
model = load_model("sign_link_model.h5")
classes = np.load("classes.npy")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    prediction_text = "Waiting..."

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])
            
            # Predict
            input_data = np.array(landmarks).reshape(1, 42, 1)
            prediction = model.predict(input_data, verbose=0)
            class_id = np.argmax(prediction)
            confidence = prediction[0][class_id]
            
            if confidence > 0.7:
                prediction_text = f"{classes[class_id]} ({int(confidence*100)}%)"

            # Visuals
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # UI Overlay
    cv2.rectangle(frame, (0,0), (300, 50), (0,0,0), -1)
    cv2.putText(frame, prediction_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Sign-Link Real-time Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
