import cv2
import numpy as np
import time
from collections import deque

# Initialize camera
cap = cv2.VideoCapture("http://172.25.165.21:4747/video")
#cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()

# Optimized HSV color range for yellow
lower_yellow = np.array([20, 100, 100])  # More inclusive lower bound
upper_yellow = np.array([40, 255, 255])  # Tighter upper bound

# Processing parameters
BLUR_SIZE = (15, 15)        # Kernel size for Gaussian blur
ERODE_ITERATIONS = 2        # Erosion iterations
DILATE_ITERATIONS = 2       # Dilation iterations
MIN_RADIUS = 2              # Minimum ball radius to detect (pixels)
MAX_RADIUS = 200            # Maximum ball radius to detect (pixels)
ASPECT_RATIO_RANGE = (0.8, 1.2)  # Width/height ratio range for circles

# Tracking variables
fps_times = deque(maxlen=60)
prev_time = time.time()
prev_center = None
stable_center = None
trail_points = deque(maxlen=10)  # For drawing motion trail

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture error")
        break

    # Pre-processing
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    blurred = cv2.GaussianBlur(frame, BLUR_SIZE, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Color thresholding
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.erode(mask, None, iterations=ERODE_ITERATIONS)
    mask = cv2.dilate(mask, None, iterations=DILATE_ITERATIONS)

    # Contour detection
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    detected_radius = 0

    if contours:
        # Filter and process contours
        valid_contours = []
        for c in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            _, (w, h), _ = cv2.minAreaRect(c)
            aspect_ratio = float(w)/h if h != 0 else 1
            
            if (MIN_RADIUS < radius < MAX_RADIUS and 
                ASPECT_RATIO_RANGE[0] < aspect_ratio < ASPECT_RATIO_RANGE[1]):
                valid_contours.append(c)

        if valid_contours:
            # Select best contour (largest that meets criteria)
            c = max(valid_contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)

            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                detected_radius = int(radius)
                
                # Only update stable center if detection is consistent
                if stable_center is None:
                    stable_center = center
                else:
                    dist = np.sqrt((center[0]-stable_center[0])**2 + (center[1]-stable_center[1])**2)
                    if dist < 50:  # Only update if movement is reasonable
                        stable_center = center
                
                trail_points.appendleft(center)

    # Draw detection results
    if center:
        # Draw motion trail
        for i in range(1, len(trail_points)):
            if trail_points[i-1] is None or trail_points[i] is None:
                continue
            thickness = int(np.sqrt(20 / float(i + 1)) * 2)
            cv2.line(frame, trail_points[i-1], trail_points[i], (0, 255, 255), thickness)
        
        # Draw current detection
        cv2.circle(frame, (int(x), int(y)), detected_radius, (0, 255, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
        # Draw stabilized center
        if stable_center:
            cv2.circle(frame, stable_center, 3, (255, 0, 0), -1)

    # Calculate and display FPS
    current_time = time.time()
    time_diff = current_time - prev_time
    prev_time = current_time
    fps = 1 / time_diff if time_diff > 0 else 0
    fps_times.append(fps)
    avg_fps = sum(fps_times) / len(fps_times)

    # Calculate speed if we have previous position
    speed = 0
    if prev_center and center:
        dx = center[0] - prev_center[0]
        dy = center[1] - prev_center[1]
        distance_px = np.sqrt(dx**2 + dy**2)
        speed = distance_px / time_diff
    prev_center = center

    # Display information
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Speed: {speed:.1f} px/s", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if center:
        cv2.putText(frame, f"Radius: {detected_radius}px", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Ball Detection", frame)
    cv2.imshow("Mask", mask)  # Show the threshold mask for debugging
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):  # Press 'c' to capture current HSV values
        print(f"Current HSV range: Lower {lower_yellow}, Upper {upper_yellow}")

cap.release()
cv2.destroyAllWindows()