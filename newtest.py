from roboflow import Roboflow
import supervision as sv
import cv2 as cv

rf = Roboflow(api_key="bEU3D1DGTYOEizYLiOKz")
project = rf.workspace().project("keyboard-detection-v2")
model = project.version(1).model

frame = cv.imread("keyboard_frame.jpg")
result = model.predict("keyboard_frame.jpg", confidence=40, overlap=30).json()
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
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label
    cv.putText(frame,
               str(label),(x1, y1 - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (36, 255, 12),
                1
                )
cv.imshow("Result", frame)
cv.waitKey(0)
cv.destroyAllWindows()

# labels = [item["class"] for item in result["predictions"]]

# detections = sv.Detections.from_roboflow(result)

# label_annotator = sv.LabelAnnotator()
# bounding_box_annotator = sv.BoxAnnotator()


# annotated_image = box_annotator.annotate(
#     scene=image, detections=detections)
# annotated_image = label_annotator.annotate(
#     scene=annotated_image, detections=detections, labels=labels)

# sv.plot_image(image=annotated_image, size=(16, 16))