import cv2

# Initialize video capture (0 for webcam, or 'video.mp4' for a video file)
cap = cv2.VideoCapture(0)

# Initialize the background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=True)

# Function to find the largest contour (assumed to be the patient's body)
def find_largest_contour(contours):
    largest_contour = max(contours, key=cv2.contourArea) if contours else None
    return largest_contour

# Variables to track movement
previous_contour_area = 100
movement_threshold = 500  # Change this based on the sensitivity required

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for background subtraction
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction
    fg_mask = background_subtractor.apply(gray_frame)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour, assumed to be the patient's body
    largest_contour = find_largest_contour(contours)

    if largest_contour is not None:
        # Calculate area of the largest contour
        contour_area = cv2.contourArea(largest_contour)

        # Draw the contour on the frame
        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

        # Check for movement by comparing contour area with previous frame
        if abs(contour_area - previous_contour_area) > movement_threshold:
            cv2.putText(frame, "Movement Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Update previous contour area
        previous_contour_area = contour_area

    # Show the frames
    cv2.imshow('Eagle View', frame)
    cv2.imshow('Foreground Mask', fg_mask)

    # Break the loop on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()