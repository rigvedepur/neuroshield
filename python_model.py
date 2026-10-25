import cv2
import torch
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model = None


def get_model():
    """Load the YOLOv8 pose model once and reuse it."""
    global _model
    if _model is None:
        _model = YOLO("yolov8n-pose.pt")
        _model.to(DEVICE)
    return _model


def pose_frame(frame):
    """Take one BGR frame and return it with pose keypoints/skeleton drawn."""
    model = get_model()
    # Resize for smooth real-time performance
    frame_resized = cv2.resize(frame, (640, 360))
    results = model(frame_resized, device=DEVICE, imgsz=640, conf=0.5, verbose=False)
    return results[0].plot(boxes=False)


def pose_stream(stream):
    """Take a stream of BGR frames and yield a stream of pose-annotated frames."""
    for frame in stream:
        yield pose_frame(frame)


def main():
    device = DEVICE
    print(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    model = get_model()

    # Open live webcam stream (0 is the default built-in webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting live webcam stream. Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from webcam.")
            break

        # Resize for smooth real-time performance
        frame_resized = cv2.resize(frame, (640, 360))

        # Run inference on the live frame
        results = model(frame_resized, device=device, imgsz=640, conf=0.5, verbose=False)
        annotated_frame = results[0].plot(boxes=False)

        # Display the live output stream in a window
        cv2.imshow("Live Webcam Pose Estimation", annotated_frame)

        # Press 'q' to quit the live stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Live stream closed.")

if __name__ == "__main__":
    main()