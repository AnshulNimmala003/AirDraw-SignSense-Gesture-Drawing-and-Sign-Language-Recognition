import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load sign language model
model = load_model("asl_model.h5")
labels = list(string.ascii_uppercase)

# Mediapipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Camera and screen setup
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()
frame_width, frame_height = 640, 480

# Modes and drawing setup
drawing_mode = False
is_eraser_mode = False
use_whiteboard = False
canvas = None
prev_x, prev_y = None, None
sign_language_text = ""
last_sign_time = 0
sign_cooldown = 1.5  # seconds

# Mode selection
def display_menu():
    print("Select Mode:")
    print("1. Gesture Control")
    print("2. Air Drawing")
    print("3. Air Drawing with Whiteboard")
    print("4. Sign Language to Text")
    return input("Enter your choice (1, 2, 3, or 4): ")

mode = display_menu()

if mode in ['2', '3']:
    canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
if mode == '3':
    use_whiteboard = True
    canvas[:] = 255  # White background

# Finger detection
def detect_number(landmarks):
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    folded = [landmarks[thumb_tip].x < landmarks[thumb_tip - 1].x]
    for tip in finger_tips:
        folded.append(landmarks[tip].y < landmarks[tip - 2].y)
    return sum(folded)

# Predict letter from hand ROI
def predict_sign_from_model(roi):
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (64, 64))
    roi = img_to_array(roi) / 255.0
    roi = np.expand_dims(roi, axis=0)
    prediction = model.predict(roi, verbose=0)[0]
    return labels[np.argmax(prediction)]

# Sign language detection
def detect_sign_language(frame, hand_landmarks):
    global sign_language_text, last_sign_time
    current_time = time.time()
    if current_time - last_sign_time < sign_cooldown:
        return

    h, w, _ = frame.shape
    x_list = [int(pt.x * w) for pt in hand_landmarks.landmark]
    y_list = [int(pt.y * h) for pt in hand_landmarks.landmark]
    x_min, x_max = max(min(x_list) - 20, 0), min(max(x_list) + 20, w)
    y_min, y_max = max(min(y_list) - 20, 0), min(max(y_list) + 20, h)

    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
        return

    try:
        letter = predict_sign_from_model(roi)
        sign_language_text += letter
        print(f"Detected Letter: {letter}")
        last_sign_time = current_time
    except Exception as e:
        print("Prediction Error:", e)

# Drawing
def handle_drawing(landmarks):
    global prev_x, prev_y, drawing_mode, is_eraser_mode
    index_tip = landmarks[8]
    fingers = detect_number(landmarks)
    if fingers == 1:
        drawing_mode = True
        is_eraser_mode = False
    elif fingers == 0:
        is_eraser_mode = True
        drawing_mode = False

    x, y = int(index_tip.x * frame_width), int(index_tip.y * frame_height)

    if drawing_mode and prev_x is not None:
        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 8)
    elif is_eraser_mode:
        eraser_color = (255, 255, 255) if use_whiteboard else (0, 0, 0)
        cv2.circle(canvas, (x, y), 20, eraser_color, -1)

    prev_x, prev_y = x, y

# Cursor movement
def handle_cursor_control(landmarks):
    index_tip = landmarks[8]
    thumb_tip = landmarks[4]
    x, y = int(index_tip.x * screen_width), int(index_tip.y * screen_height)
    pyautogui.moveTo(x, y)
    distance = np.linalg.norm(np.array([index_tip.x, index_tip.y]) - np.array([thumb_tip.x, thumb_tip.y]))
    if distance < 0.05:
        pyautogui.click()

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if canvas is None and mode in ['2', '3']:
        canvas = np.zeros_like(frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if mode == '1':
                handle_cursor_control(hand_landmarks.landmark)
            elif mode in ['2', '3']:
                handle_drawing(hand_landmarks.landmark)
            elif mode == '4':
                detect_sign_language(frame, hand_landmarks)

    if mode in ['2', '3']:
        if use_whiteboard:
            blended = cv2.addWeighted(canvas, 0.7, frame, 0.3, 0)
            frame = blended
        else:
            frame = cv2.add(frame, canvas)

    if mode == '4':
        cv2.putText(frame, f"Text: {sign_language_text}", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Based Interface", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
