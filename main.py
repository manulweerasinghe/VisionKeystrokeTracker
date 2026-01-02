from inference_sdk import InferenceHTTPClient
import cv2 as cv
import mediapipe as mp
import numpy as np

# File Names
keyboaed_fileName = "keyboard_frame.png"
keyboaed_fileName_jpeg = "keyboard_frame01.jpg"
detected_filename = "detected_alpha.png"
# Data
setup_window = "Setup"
cam_index = -1
max_points = 4
matrix = None
fingertips = [4, 8, 12, 16, 20]
key_positions = []
key_labels = []
hand_positions = []
static_positions = []
src_points = []
handedness = []
# Switcher
enable_annote = False
enable_handtracking = False
enable_overlay = True
_get_overlay = False
warp_ready = False
set_warp = True
enable_preview = True

def keyDetector(cam_index): # Detect the keyboard
    global output_size, matrix, w, h, warp_ready

    cap = cv.VideoCapture(cam_index, cv.CAP_DSHOW) # Start capture with directShow
    ret, frame = cap.read() # read a frame
    if not ret: # if there is no frame
        cap.release()
        exit()
    h, w, _ = frame.shape
    cap.release() # stop the capture
    if warp_ready:
        frame = cv.warpPerspective(frame, matrix, (w, h))
    cv.imwrite(keyboaed_fileName, frame) # saving the frame to a jpeg

    # initilize the model
    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="bEU3D1DGTYOEizYLiOKz"
    )
    keyResult = CLIENT.infer("keyboard_frame.jpg", model_id="keyboard-detection-v2/1")

    return keyResult, h, w

def seperator(result, h, w): # Calculate bounds and annote to a image
    global _get_overlay, key_positions, static_positions, key_labels
    bgra_image = np.zeros((h, w, 4), dtype=np.uint8)
    # bgra_image = cv.imread(keyboaed_fileName)

    key_labels.clear()
    key_positions.clear()
    # overhead = 0
    for pred in result["predictions"]:
        # Break down json format
        x, y = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]
        label = pred["class"]
        # print(label, x, y)
        # y += int(overhead/2)

        # Get top-left & bottom-right
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        print(label, x, y)
        cv.rectangle(bgra_image, (x1, y1), (x2, y2), (0, 255, 0, 128), 2)
        # Draw label
        cv.putText(bgra_image,
                str(label),(x1, y1 + 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 0, 255, 128),
                    1)
        if label=="keyboard":
            continue
        key_labels.append(label)
        key_positions.append(list((int(x), int(y))))
    cv.imwrite(detected_filename, bgra_image)
    static_positions = np.array(key_positions)
    key_labels = np.array(key_labels)
    _get_overlay = True

def startHandTrack(): # Initialize MediaPipe Hands
    global hands, mp_drawing, mp_drawing_styles, mp_hands

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        max_num_hands = 2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

def handTrack(frame, h, w): # Tracks the hands every frame
    global hands
    key_caps, handedness = None, None
    # To improve the proformance
    frame.flags.writeable = False
    results = hands.process(frame)
    if results.multi_hand_landmarks:
        key_caps, handedness = getDistance(results, h, w)

    if enable_annote:
        frame.flags.writeable = True
        frame = handAnnotator(frame, results, h, w)
        # frame = frame[:, :, np.newaxis]
    return frame, key_caps, handedness
    
def handAnnotator(frame, results, h, w): # Annote the hands every frame
    global ft, mp_hands, mp_drawing, mp_drawing_styles

    # Draw the hand annotations on the image.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # for ft in fingertips:
                # lm = hand_landmarks.landmark[ft]
                # cx, cy = int(lm.x * w), int(lm.y * h)
                # print(cx, cy)
                # cv.circle(frame, (cx, cy), 10, (255, 0, 0, 255), cv.FILLED)
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    return frame   

def mouseCallback(event, x, y, flags, param): # Look for mouse click in cv
    global src_points, warp_ready

    if event == cv.EVENT_LBUTTONDOWN and len(src_points) < max_points:
        src_points.append([x, y])
        print(f"Point {len(src_points)}: ({x}, {y})")

        if len(src_points) == max_points:
            warp_ready = True
            print("✅ All 4 points selected. Warping ready.")

def setup(event): # Setting up the camera feed
    global cam_index, src_points, warp_ready, set_warp, matrix, output_size, setup_window

    cam_index += 1
    cv.destroyAllWindows()
    cap = cv.VideoCapture(cam_index, cv.CAP_DSHOW)
    cv.namedWindow(setup_window)
    cv.setMouseCallback(setup_window, mouseCallback)

    # Get hight, width of frame
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture webcam frame")
    h, w, _ = frame.shape
    print(h, w)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display_frame = frame.copy()

        # Draw selected points
        for pt in src_points:
            cv.circle(display_frame, tuple(pt), 5, (0, 255, 0), -1)
        # If 4 points are selected, apply warp
        if warp_ready:
            if set_warp:
                set_warp = False
                cv.destroyAllWindows()
            pts1 = np.array(src_points, dtype=np.float32)
            pts2 = np.array([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ], dtype=np.float32)

            matrix = cv.getPerspectiveTransform(pts1, pts2)
            warped = cv.warpPerspective(frame, matrix, (w, h))
            cv.imshow(f"{setup_window} (Completed)", warped)
        else:
            cv.imshow(setup_window, display_frame)
        key = cv.waitKey(1)
        
        if key == ord('r'):  # Reset points
            src_points = []
            warp_ready = False
            print("🔄 Reset points.")
            set_warp = True
            setup(None)
        elif key & 0xFF == 27:  # ESC to exit
            cap.release()
            cv.destroyAllWindows()
            break
 
def show(): # Display the camera feed with keyboard annotation and hand annotations
    global cam_index, matrix, enable_preview
    cv.destroyAllWindows()
    if _get_overlay:
        overlay = cv.imread(detected_filename, cv.IMREAD_UNCHANGED)
        overlay_bgr = overlay[:, :, :3]
        overlay_alpha = overlay[:, :, 3:] / 255.0 
        overlay_value = overlay_alpha * overlay_bgr + (1 - overlay_alpha)
    cap = cv.VideoCapture(cam_index, cv.CAP_DSHOW)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture webcam frame")
    h, w, _ = frame.shape
    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to capture webcam frame")
        if warp_ready:
            frame = cv.warpPerspective(frame, matrix, (w, h))

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        if enable_overlay and _get_overlay:
            frame = (overlay_value * frame).astype(np.uint8)
        
        frame, _, _ = handTrack(frame, h, w)
        
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        cv.imshow("Hand Tracking", frame) 
        if cv.waitKey(1) & 0xFF == 27: # Exit when press "q"
            break
    enable_preview = True
    cap.release()
    cv.destroyAllWindows()

def getDistance(hand_result, h, w): # Gets the distance between fingertips and key centers
    global ft, key_labels, hand_positions, static_positions

    handedness.clear()
    hand_positions.clear()
    if hand_result.multi_hand_landmarks:
        # print(hand_result.multi_handedness)
        for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
            handedness.append(hand_result.multi_handedness[idx].classification[0].label)
            # score = hand_result.multi_handedness[idx].classification[0].score
            # print(f"Hand {idx+1}: {handedness} (confidence: {score:.2f})")
            for ft in fingertips:
                lm = hand_landmarks.landmark[ft]
                hand_positions.append(list((int(lm.x*w), int(lm.y*h))))

        dynamic_positions = np.array(hand_positions)
        distances = np.linalg.norm(static_positions[:, None, :] - dynamic_positions[None, :, :], axis=2)
        min_distances = np.min(distances, axis=0)  # Closest static point to each dynamic point
        closest_static_indices = np.argmin(distances, axis=0)
        # print(min_distances,"::::",key_labels[closest_static_indices])
        return key_labels[closest_static_indices], handedness

def oneFrame(cap, w, h):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture webcam frame")
    if warp_ready:
        frame = cv.warpPerspective(frame, matrix, (w, h))
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame, key_caps, handedness  = handTrack(frame, h, w)
    print(key_caps, handedness)

# startHandTrack()
# show()
# setup() # this need to run first
# # cv.destroyAllWindows()
# keyboard_result, height, width = keyDetector(cam_index)
# seperator(keyboard_result, height, width)
# show()
# time.sleep(5.00)



# run keyboarddector get list of predictions
# run handraker and take values
# calculate the distance between the key and finger
# output key to the related finger