import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize mediapipe hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Function to map coordinates
def map_coordinates(x, y, frame_width, frame_height, screen_width, screen_height):
    screen_x = np.interp(x, [0, frame_width], [0, screen_width])
    screen_y = np.interp(y, [0, frame_height], [0, screen_height])
    return screen_x, screen_y

# Function to smooth cursor movement
def smooth_coordinates(new_x, new_y, old_x, old_y, alpha=0.8):
    smoothed_x = alpha * new_x + (1 - alpha) * old_x
    smoothed_y = alpha * new_y + (1 - alpha) * old_y
    return smoothed_x, smoothed_y

# Function to check if movement exceeds the threshold
def is_movement_significant(new_x, new_y, old_x, old_y, threshold=5):
    distance = np.sqrt((new_x - old_x)**2 + (new_y - old_y)**2)
    return distance > threshold

# Get screen size
screen_width, screen_height = pyautogui.size()

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Initialize previous coordinates for smoothing
prev_screen_x, prev_screen_y = pyautogui.position()  # Start with current mouse position
dragging = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get coordinates of middle finger tip (landmark 12) for cursor movement
            middle_x = int(hand_landmarks.landmark[12].x * frame_width)
            middle_y = int(hand_landmarks.landmark[12].y * frame_height)
            screen_x, screen_y = map_coordinates(middle_x, middle_y, frame_width, frame_height, screen_width, screen_height)

            # Smooth the cursor movement
            screen_x, screen_y = smooth_coordinates(screen_x, screen_y, prev_screen_x, prev_screen_y)

            # Move the mouse cursor if the movement is significant
            if is_movement_significant(screen_x, screen_y, prev_screen_x, prev_screen_y):
                pyautogui.moveTo(screen_x, screen_y)
                prev_screen_x, prev_screen_y = screen_x, screen_y  # Update only if the movement is significant

            # Perform click and drag if index finger and thumb are close
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

            # Print debug information
            print(f"Distance between thumb and index finger: {distance}")
            
            if distance < 0.03:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
                    print("Mouse Down")
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                    print("Mouse Up")

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
