# Yellow-Ball-Detection
Detection of a yellow smiley ball using CV2.

**Overview**

This project implements real-time yellow ball detection using computer vision techniques. The system captures video feed, processes each frame to detect yellow objects, tracks their movement, and provides visual feedback along with performance metrics.

**Features**

    •	Real-time yellow object detection using HSV color thresholding

    •	Advanced contour processing with size and shape filtering

    •	Motion tracking with stabilization

    •	Visual feedback including:

        o	Detected ball outline
        
        o	Center point
        
        o	Motion trail
        
        o	Stabilized center point

    •	Performance metrics:

        o	Real-time FPS display
        
        o	Object speed calculation (in pixels/second)
        
        o	Object radius measurement


**Requirements**

    •	Python 3.x

    •	OpenCV (cv2)
    
    •	NumPy
    
    •	Collections.deque


**Installation**

    1.	Clone the repository:

            git clone https://github.com/yourusername/Yellow-Ball-Detection.git
            cd Yellow-Ball-Detection

    2.	Install the required packages:

            pip install opencv-python numpy


**Usage**

    1.	Run the detection script:

            python yellow_ball_detection.py
    
    2.	Controls:
    
            o	Press 'q' to quit the application
    
            o	Press 'c' to print current HSV threshold values to console
    
    3.	Camera Configuration:
    
            o	By default, the script connects to an IP camera at http://172.25.165.21:4747/video
    
            o	To use a local webcam, uncomment cap = cv2.VideoCapture(0) and comment out the IP camera line

**Detection**

![{D82BCD84-6CB4-4C47-B1F3-E25364E3557E}](https://github.com/user-attachments/assets/d93d3cb5-01ca-4ca6-bec3-e6e831997189)


**Configuration**

    Adjust these parameters in the code for optimal performance:
    
    # HSV color range for yellow detection
    
    lower_yellow = np.array([20, 100, 100])
    
    upper_yellow = np.array([40, 255, 255])

    
    # Processing parameters
    
    BLUR_SIZE = (15, 15)        # Gaussian blur kernel size
    
    ERODE_ITERATIONS = 2        # Erosion iterations
    
    DILATE_ITERATIONS = 2       # Dilation iterations
    
    MIN_RADIUS = 2              # Minimum detection radius (pixels)
    
    MAX_RADIUS = 200            # Maximum detection radius (pixels)

    ASPECT_RATIO_RANGE = (0.8, 1.2)  # Valid width/height ratio range


**Performance Notes**

    •	The system maintains an average FPS display for performance monitoring

    •	Processing includes multiple optimization steps:
      
        o	Frame resizing
      
        o	Gaussian blur for noise reduction
      
        o	Morphological operations (erosion/dilation)
      
        o	Contour filtering by size and shape


**Troubleshooting**
   
    1.	Camera not opening:
   
        o	Check camera connection
   
        o	Verify correct camera index or IP address
   
        o	Ensure no other application is using the camera
   
    2.	Poor detection:
   
        o	Adjust HSV thresholds using the 'c' command to print current values
   
        o	Modify blur size or morphological operation iterations
   
        o	Check lighting conditions (HSV is lighting-sensitive)
   
    3.	High latency:
   
        o	Reduce frame resolution
   
        o	Decrease blur kernel size
   
        o	Use fewer contour processing steps


**License**

    This project is open-source and available under the MIT License.
    Future Improvements

    •	Add calibration mode for automatic HSV range detection

    •	Implement 3D position estimation using stereo vision

    •	Add serial communication for robotics applications

    •	Develop a graphical interface for parameter tuning


**Acknowledgments**

    •	OpenCV community for excellent computer vision libraries

    •	NumPy for efficient array operations

