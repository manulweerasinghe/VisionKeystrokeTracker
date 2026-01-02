# from roboflow import Roboflow
# import supervision as sv
# import cv2 as cv
# import numpy as np

print(graph.get_input_devices())
# # Roboflow setup
# rf = Roboflow(api_key="bEU3D1DGTYOEizYLiOKz")
# project = rf.workspace().project("keyboard-detection-v2")
# model = project.version(1).model

# # Run inference
# result = model.predict("keyboard_frame01.jpg", confidence=40, overlap=30).json()
# print(result, type(result))
# # Extract data from Roboflow response
# detections = sv.Detections(
#     xyxy=np.array([
#         [prediction["x"] - prediction["width"] / 2,
#          prediction["y"] - prediction["height"] / 2,
#          prediction["x"] + prediction["width"] / 2,
#          prediction["y"] + prediction["height"] / 2]
#         for prediction in result["predictions"]
#     ]),
#     class_id=np.array([
#         prediction["class"] for prediction in result["predictions"]
#     ]),
#     confidence=np.array([
#         prediction["confidence"] for prediction in result["predictions"]
#     ])
# )

# # Load and annotate image
# image = cv.imread("keyboard_img.jpg")  # Use the same image used for prediction
# bounding_box_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# image = bounding_box_annotator.annotate(scene=image, detections=detections)
# image = label_annotator.annotate(scene=image, detections=detections)

# # Show the image
# sv.plot_image(image=image, size=(16, 16))