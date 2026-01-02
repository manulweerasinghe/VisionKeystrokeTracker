import mediapipe as mp
import cv2 as cv
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
frame_global = None

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]
# Create a hand landmarker instance with the live stream mode:
def print_result(result, output_image, timestamp_ms):
    print('hand landmarker result: {}'.format(result))
    # global frame_global
    # frame = frame_global.copy()
    # h, w, _ = frame.shape()

    # if result.hand_landmarks:
    #     for hand_landmarks in result.hand_landmarks:
    #         points = []
    #         for lm in hand_landmarks:
    #             cx, cy = int(lm.x * w), int(lm.y * h)
    #             points.append((cx, cy))
    #             cv.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    #         for start_idx, end_idx in HAND_CONNECTIONS:
    #             cv.line(frame, points[start_idx], points[end_idx], (255, 0, 0), 2)

    # cv.imshow("Hand Tracking (detect_async)", frame)


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands = 2)
detector = HandLandmarker.create_from_options(options)

# Capture the video feed with directShow
cap = cv.VideoCapture(2, cv.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open the camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Didn't read a frame")
        break
    mp_image = mp.Image(
        image_format = mp.ImageFormat.SRGB,
        data = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    )

    frame_global = frame.copy()
    timestamp = int(time.time() * 1000)
    result = detector.detect_async(mp_image, timestamp)

    if cv.waitKey(1) == ord('q'):
            break

     
# When every thing is finished
cap.release()
cv.destroyAllWindows()