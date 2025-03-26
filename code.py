import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import string

# Initialize Mediapipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Capture webcam
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

# Variables
drawing_mode = False
prev_x, prev_y = None, None
canvas = None
use_whiteboard = False
is_eraser_mode = False
number_gesture_start_time = None
number_sequence = []
last_operator_time = 0
operator_cooldown = 2  # 2 seconds cooldown
sign_language_text = ""
last_sign_time = 0
sign_cooldown = 2

# Sign Language Mapping (A-Z using basic gestures)
sign_language_mapping = {i: letter for i, letter in enumerate(string.ascii_uppercase, 1)}

def display_menu():
    print("Select Mode:")
    print("1. Gesture Control")
    print("2. Air Drawing")
    print("3. Air Drawing with Whiteboard")
    print("4. Basic Math using Gestures")
    print("5. Sign Language to Text")
    mode = input("Enter your choice (1, 2, 3, 4, or 5): ")
    return mode

mode = display_menu()

if mode == '3':
    use_whiteboard = True
    canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255

# Function to recognize numbers using finger count
def detect_number(landmarks):
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    finger_folded = []

    if landmarks[thumb_tip].x < landmarks[thumb_tip - 1].x:
        finger_folded.append(1)
    else:
        finger_folded.append(0)

    for tip in finger_tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            finger_folded.append(1)
        else:
            finger_folded.append(0)

    return sum(finger_folded)

# Function to detect sign language gestures
def detect_sign_language(landmarks):
    global last_sign_time, sign_language_text
    current_time = time.time()

    if current_time - last_sign_time < sign_cooldown:
        return“AirDraw SignSense: Gesture Drawing and Sign Language Recognition”

    fingers_count = detect_number(landmarks)
    if fingers_count in sign_language_mapping:
        letter = sign_language_mapping[fingers_count]
        sign_language_text += letter
        print(f"Detected Letter: {letter}")
        print(f"Current Text: {sign_language_text}")
        last_sign_time = current_time

# Function to handle drawing and erasing
def handle_drawing(landmarks):
    global prev_x, prev_y, is_eraser_mode, drawing_mode
    index_tip = landmarks[8]

    # Check for the drawing gesture (Index finger up and others down)
    fingers_count = detect_number(landmarks)
    if fingers_count == 1:
        drawing_mode = True
        is_eraser_mode = False
    
    # Check for the eraser gesture (Closed fist)
    elif fingers_count == 0:
        is_eraser_mode = True
        drawing_mode = False

    # Drawing or erasing
    x, y = int(index_tip.x * 640), int(index_tip.y * 480)
    if drawing_mode and prev_x is not None and prev_y is not None:
        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 8)
    elif is_eraser_mode and prev_x is not None and prev_y is not None:
        cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 40)
    prev_x, prev_y = x, y

# Function for gesture-based cursor control
def handle_cursor_control(landmarks):
    index_tip = landmarks[8]
    thumb_tip = landmarks[4]

    # Calculate finger positions for cursor movement
    x, y = int(index_tip.x * screen_width), int(index_tip.y * screen_height)
    pyautogui.moveTo(x, y)

    # Detect pinch gesture for mouse click
    distance = np.linalg.norm(np.array([index_tip.x, index_tip.y]) - np.array([thumb_tip.x, thumb_tip.y]))
    if distance < 0.05:
        pyautogui.click()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if canvas is None and not use_whiteboard:
        canvas = np.zeros_like(frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            if mode == '1':
                handle_cursor_control(landmarks)
            elif mode == '2' or mode == '3':
                handle_drawing(landmarks)
            elif mode == '5':
                detect_sign_language(landmarks)

    if use_whiteboard:
        frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    else:
        frame = cv2.add(canvas, frame)

    if mode == '5':
        cv2.putText(frame, f"Text: {sign_language_text}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    cv2.imshow('Hand Gesture Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
