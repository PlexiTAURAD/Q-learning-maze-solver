import cv2
import numpy as np

# Function to apply a kernel to an image
def apply_kernel(image, kernel):
    return cv2.filter2D(image, -1, kernel)

# Function to display a section of the image as a normalized pixel matrix
def display_normalized_matrix(image, top_left, size=(5, 5)):
    # Extract a section of the image
    x, y = top_left
    h, w = size
    section = image[x:x+h, y:y+w]

    # Normalize pixel values to [0, 1]
    normalized_section = section / 255.0

    # Create a blank image to display the matrix
    matrix_display = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.putText(matrix_display, "Normalized Pixel Matrix", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Add the normalized values as text to the display
    for i in range(h):
        for j in range(w):
            value = f"{normalized_section[i, j]:.2f}"
            cv2.putText(matrix_display, value, (j * 60 + 20, i * 40 + 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return matrix_display

# Define kernels for edge, line, and point detection
edge_kernel = np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]])

line_kernel = np.array([[ 0, -1,  0],
                        [-1,  4, -1],
                        [ 0, -1,  0]])

point_kernel = np.array([[-1, -1, -1],
                         [-1,  9, -1],
                         [-1, -1, -1]])

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'c' to capture an image or 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Display the live webcam feed
    cv2.imshow("Webcam Feed", frame)

    # Wait for user input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Capture image
        print("Image captured!")
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()

# Convert the captured frame to grayscale
gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Apply kernels to the grayscale image
edge_detection = apply_kernel(gray_image, edge_kernel)
line_detection = apply_kernel(gray_image, line_kernel)
point_detection = apply_kernel(gray_image, point_kernel)

# Define the top-left corner and size of the section to extract
top_left = (50, 50)  # Top-left corner of the section
section_size = (5, 5)  # Size of the section (5x5 pixels)

# Get the normalized pixel matrix display
matrix_display = display_normalized_matrix(gray_image, top_left, section_size)

# Display the original grayscale image, processed images, and the matrix display
cv2.imshow("Original Grayscale Image", gray_image)
cv2.imshow("Edge Detection", edge_detection)
cv2.imshow("Line Detection", line_detection)
cv2.imshow("Point Detection", point_detection)
cv2.imshow("Normalized Pixel Matrix", matrix_display)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()