import cv2

video_path = r"C:\Users\murat\Desktop\StajHW\LineLogic\New Videos 2\IMG_0015_blurred.MOV"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Failed to open video.")
else:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video resolution: {width} x {height}")
