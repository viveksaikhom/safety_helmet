import onnxruntime as ort
import cv2
import numpy as np
import RPi.GPIO as ti_gpio

ti_gpio.setmode(ti_gpio.BOARD)
RED_PIN = 16
GREEN_PIN = 18
BLUE_PIN = 22
ti_gpio.setup(RED_PIN, ti_gpio.OUT)
ti_gpio.setup(GREEN_PIN, ti_gpio.OUT)
ti_gpio.setup(BLUE_PIN, ti_gpio.OUT)


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

# Load your model with onnxruntime-tidl
model_path = '/path/to/your/model.onnx'  # Path to your ONNX model
providers = ['TIDLExecutionProvider', 'TIDLCompilationProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)

# Prepare input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Capture video from the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def preprocess_image(image, input_shape):
    image_resized = cv2.resize(image, (input_shape[2], input_shape[3]))  # Resize to the expected input size
    image_normalized = image_resized.astype(np.float32) / 255.0  # Normalize pixel values
    image_transposed = np.transpose(image_normalized, (2, 0, 1))  # Convert to CHW format
    image_batch = np.expand_dims(image_transposed, axis=0)  # Add batch dimension
    return image_batch

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Preprocess the frame to match the model's input format
    input_data = preprocess_image(frame, session.get_inputs()[0].shape)

    # Run inference on the model
    outputs = session.run([output_name], {input_name: input_data})
    boxes = outputs[0]

    # Check the result and set LED colors
    label = "No object"
    if boxes[0][5] == "with_helmet":
        label = "With Helmet"
        set_led_color("green")
    elif boxes[0][5] == "without_helmet":
        label = "Without Helmet"
        set_led_color("red")

    # Draw bounding boxes and label on the frame
    cv2.rectangle(frame, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (0, 255, 0), 2)
    cv2.putText(frame, label, (int(boxes[0][0]), int(boxes[0][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Helmet Detection', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ti_gpio.cleanup()
