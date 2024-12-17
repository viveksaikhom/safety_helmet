import cv2

camera = cv2.VideoCapture(0)  # USB camera 0

if not camera.isOpened():
    print("Error: Unable to access the camera.")
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press 'q' to quit.")
while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
