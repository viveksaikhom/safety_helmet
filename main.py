import cv2
import onnxruntime as ort
import numpy as np
import time

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

user = input("Should we start? (y/n): ")
if user == "y":
    print('Starting...')
else:
    exit()

tidl_providers = [
    (
        "TIDLExecutionProvider",
        {
            "tidl_network_path": "/opt/model_zoo/20241208-173443_yolox_nano_lite_onnxrt_AM62A/artifacts/detslabels_tidl_net.bin",
            "tidl_input_output_path": "/opt/model_zoo/20241208-173443_yolox_nano_lite_onnxrt_AM62A/artifacts/detslabels_tidl_io_1.bin",
            "tidl_delegate_path": "/opt/ti/tidl/libs/libtidl_delegate.so",
        },
    )
]

onnx_model_path = "/opt/model_zoo/20241208-173443_yolox_nano_lite_onnxrt_AM62A/model/model.onnx"
session = ort.InferenceSession(onnx_model_path, providers=tidl_providers)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def preprocess_image(image, input_size=(416, 416)):
    resized_image = cv2.resize(image, input_size)
    normalized_image = resized_image / 255.0
    input_tensor = np.transpose(normalized_image, (2, 0, 1)).astype('float32')
    return np.expand_dims(input_tensor, axis=0)

def postprocess(output, conf_threshold=0.5):
    boxes, scores, labels = output
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    class_names = ["without_helmet", "with_helmet"]

    detections = []
    for i in range(len(boxes)):
        class_id = int(labels[i])
        confidence = scores[i]
        label = class_names[class_id]
        detections.append(f"{label}: {confidence:.2f}")
        print(f"Detected {label} with confidence {confidence:.2f}")

    return detections

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    input_tensor = preprocess_image(frame)
    outputs = session.run([output_name], {input_name: input_tensor})

    detections = postprocess(outputs[0])

    current_time = time.time()
    if current_time - last_time >= 1:
        print(f"\nDetection results at {current_time:.2f} seconds:")
        for detection in detections:
            print(detection)
        last_time = current_time

    for i in range(len(detections)):
        cv2.putText(frame, detections[i], (10, (i + 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOX Detection - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
