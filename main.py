import time
import cv2
import onnxruntime as ort
import numpy as np
import RPi.GPIO as ti_gpio

print('''
            __      _         
 ___  __ _ / _| ___| |_ _   _ 
/ __|/ _` | |_ / _ \ __| | | |
\__ \ (_| |  _|  __/ |_| |_| |
|___/\__,_|_|  \___|\__|\__, |
__     ______           |___/ 
\ \   / / ___|                
 \ \ / /\___ \                
  \ V /  ___) |               
   \_/  |____/                
''')

print("Note:\nPIN 16: RED, PIN 18:GREEN, PIN 22:BLUE")
user = input("Should we start? (y/n): ")
if user == "y":
    print('Starting...')
else:
    exit()

ti_gpio.setmode(ti_gpio.BOARD)
RED_PIN = 16
GREEN_PIN = 18
BLUE_PIN = 22
ti_gpio.setup(RED_PIN, ti_gpio.OUT)
ti_gpio.setup(GREEN_PIN, ti_gpio.OUT)
ti_gpio.setup(BLUE_PIN, ti_gpio.OUT)

model_path = '/opt/model_zoo/20241208-173443_yolox_nano_lite_onnxrt_AM62A/model/model.onnx'
session = ort.InferenceSession(model_path)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def preprocess_image(image, input_shape):
    image_resized = cv2.resize(image, (input_shape[2], input_shape[3]))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    image_batch = np.expand_dims(image_transposed, axis=0)
    return image_batch


def set_led_color(color):
    if color == "green":
        ti_gpio.output(RED_PIN, ti_gpio.LOW)
        ti_gpio.output(GREEN_PIN, ti_gpio.HIGH)
        ti_gpio.output(BLUE_PIN, ti_gpio.LOW)
    elif color == "blue":
        ti_gpio.output(RED_PIN, ti_gpio.LOW)
        ti_gpio.output(GREEN_PIN, ti_gpio.LOW)
        ti_gpio.output(BLUE_PIN, ti_gpio.HIGH)
    elif color == "red":
        ti_gpio.output(RED_PIN, ti_gpio.HIGH)
        ti_gpio.output(GREEN_PIN, ti_gpio.LOW)
        ti_gpio.output(BLUE_PIN, ti_gpio.LOW)


while True:
    ret, frame = cap.read()

    # If frame was captured successfully
    if not ret:
        print("Failed to capture image")
        break

    input_data = preprocess_image(frame, session.get_inputs()[0].shape)

    outputs = session.run([output_name], {input_name: input_data})

    boxes = outputs[0]

    label = "No object"

    if boxes[0][5] == "with helmet":
        label = "With Helmet"
        set_led_color("green")
    elif boxes[0][5] == "without_helmet":
        label = "Without Helmet"
        set_led_color("red")

    cv2.rectangle(frame, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (0, 255, 0), 2)
    cv2.putText(frame, label, (int(boxes[0][0]), int(boxes[0][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if label == "With Helmet":
        print("Helmet detected")

    cv2.imshow('Helmet Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ti_gpio.cleanup()
