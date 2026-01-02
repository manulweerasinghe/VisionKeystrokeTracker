import cv2 as cv
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands = 2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize Webcam
cap = cv.VideoCapture(3, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

fingertips = [4, 8, 12, 16, 20]
setDementions = False
# Loop
hand_positions = []
while True:
    ret, frame = cap.read() # Capture the aviable frame
    if not ret:
        print("frame didn't capture")
        break

    # To improve the proformance
    frame.flags.writeable = False
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frame)
    # print(type(results))
    # print(results)

    #Get dementions once
    if not setDementions:
        h, w, _ = frame.shape
        print(h,w)
        setDementions = True

    # Draw the hand annotations on the image.
    frame.flags.writeable = True
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for ft in fingertips:
                lm = hand_landmarks.landmark[ft]
                hand_positions.extend(list((lm.x*w, lm.y*h)))
                
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv.circle(frame, (cx, cy), 10, (0, 255, 0), cv.FILLED)
            # mp_drawing.draw_landmarks(
            #     frame,
            #     hand_landmarks,
            #     mp_hands.HAND_CONNECTIONS
            #     )
    # print(hand_positions)
    # Our operations on the frame come here
    # color = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
    # Display the resulting frame
    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'): # Exit when press "q"
        break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()