qimport cv2
import mediapipe as mp
import autopy

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize the video capture device
cap = cv2.VideoCapture(0)

# Initialize the hand tracking model
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    # Create a window to display the video
    cv2.namedWindow('AI Virtual Mouse', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('AI Virtual Mouse', 640, 480)
    
    # Variable to track if left mouse button is pressed
    is_left_button_pressed = False

    while True:
        # Read the frame from the video capture device
        ret, frame = cap.read()
        
        # Detect the hand landmarks
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # If a hand is detected
        if results.multi_hand_landmarks:
            # Get the landmarks for the first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get the coordinates of the index finger
            index_finger_coords = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_coords.x * frame.shape[1]), int(index_finger_coords.y * frame.shape[0])
            
            # Move the mouse cursor to the index finger coordinates
            autopy.mouse.move(x, y)
            
            # Check if the index finger is extended
            index_finger_extended = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
            
            # Check if the middle finger is extended
            middle_finger_extended = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
            
            # Perform left-click action if index and middle fingers are extended
            if index_finger_extended and middle_finger_extended:
                if not is_left_button_pressed:
                    autopy.mouse.toggle(autopy.mouse.Button.LEFT, True)
                    is_left_button_pressed = True
            else:
                if is_left_button_pressed:
                    autopy.mouse.toggle(autopy.mouse.Button.LEFT, False)
                    is_left_button_pressed = False
            
            # Check if the ring finger is extended
            ring_finger_extended = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
            
            # Perform right-click action if ring finger is extended
            if ring_finger_extended:
                autopy.mouse.toggle(autopy.mouse.Button.RIGHT, True)
            else:
                autopy.mouse.toggle(autopy.mouse.Button.RIGHT, False)
            
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display the frame
        cv2.imshow('AI Virtual Mouse', frame)
        
        # If the 'q' key is pressed, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the
