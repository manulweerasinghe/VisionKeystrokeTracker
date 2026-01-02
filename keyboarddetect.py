from roboflow import Roboflow
# import supervision as sv
import cv2 as cv
import numpy as np
# rf = Roboflow(api_key="bEU3D1DGTYOEizYLiOKz")
# project = rf.workspace().project("keyboard-detection-v2")
# model = project.version(1).model

from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="bEU3D1DGTYOEizYLiOKz"
)

# cap = cv.VideoCapture(3, cv.CAP_DSHOW)
# ret, frame = cap.read()
# if not ret:
#     exit()
# cap.release()
# cv.imwrite("keyboard_frame.jpg", frame)
image = cv.imread("keyboard_frame.png")

result = CLIENT.infer("keyboard_frame.png", model_id="keyboard-detection-v2/1")
# result = model.predict("keyboard_frame.jpg", confidence=40, overlap=30).json()


# print(result,"\n",type(result))

for pred in result["predictions"]:
    # Break down json format
    x, y = pred["x"], pred["y"]
    w, h = pred["width"], pred["height"]
    label = pred["class"]
    if label == "keyboard":
        print(y,h)

    # Get top-left & bottom-right
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    # Draw rectangle
    cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label
    cv.putText(image,
               str(label),(x1, y1 - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (36, 255, 12),
                1
                )
cv.imshow("Result", image)
cv.waitKey(0)
cv.destroyAllWindows()
# labels = [item["class"] for item in result["predictions"]]
# print(len(labels))
# detections = sv.Detections.from_roboflow(result)
# detections = sv.Detections.from_roboflow(result)

# label_annotator = sv.LabelAnnotator()
# bounding_box_annotator = sv.BoxAnnotator()

# image = cv.imread("your_image.jpg")

# annotated_image = label_annotator.annotate(
#     scene=image, detections=detections)
# annotated_image = label_annotator.annotate(
#     scene=annotated_image, detections=detections, labels=labels)

# sv.plot_image(image=annotated_image, size=(16, 16))
