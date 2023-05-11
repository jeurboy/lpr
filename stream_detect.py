from imageai.Detection import VideoObjectDetection
import os
import cv2

# print(cv2.getBuildInformation())

execution_path = os.getcwd()

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 30)

detector = VideoObjectDetection()
detector.useCPU()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "model/yolov3.pt"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(camera_input=camera,
                                             output_file_path=os.path.join(execution_path, "camera_detected_video"), frames_per_second=20, log_progress=True, minimum_percentage_probability=30)

# print(video_path)
