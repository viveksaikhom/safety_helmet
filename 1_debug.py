import cv2
import onnxruntime as ort
import numpy as np
import RPi.GPIO as ti_gpio
import pdb

# Cleanup previous GPIO settings
ti_gpio.cleanup()

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

# Explicitly setting the providers to resolve the ValueError
providers = ['TIDLExecutionProvider', 'TIDLCompilationProvider', 'CPUExecutionProvider']
try:
    session = ort.InferenceSession(model_path, providers=providers)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Debugging: Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()


def preprocess_image(image, input_shape):
    try:
        image_resized = cv2.resize(image, (input_shape[2], input_shape[3]))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_transposed = np.transpose(image_normalized, (2, 0, 1))
        image_batch = np.expand_dims(image_transposed, axis=0)
        return image_batch
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None


def set_led_color(color):
    try:
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
    except Exception as e:
        print(f"Error while setting LED color: {e}")


while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        continue

    input_data = preprocess_image(frame, session.get_inputs()[0].shape)

    if input_data is None:
        print("Skipping frame due to preprocessing error.")
        continue

    try:
        outputs = session.run([output_name], {input_name: input_data})
        boxes = outputs[0]
        print(f"Output boxes: {boxes}")
    except Exception as e:
        print(f"Error during inference: {e}")
        continue

    label = "No object"

    if len(boxes) > 0 and boxes[0][5] == "with helmet":
        label = "With Helmet"
        set_led_color("green")
    elif len(boxes) > 0 and boxes[0][5] == "without_helmet":
        label = "Without Helmet"
        set_led_color("red")

    print(f"Detected label: {label}")

    if label == "With Helmet":
        print("Helmet detected")

    cv2.rectangle(frame, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (0, 255, 0), 2)
    cv2.putText(frame, label, (int(boxes[0][0]), int(boxes[0][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Helmet Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program.")
        break

cap.release()
cv2.destroyAllWindows()
ti_gpio.cleanup()
