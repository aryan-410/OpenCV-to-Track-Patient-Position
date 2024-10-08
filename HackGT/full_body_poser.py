import math
import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Initialize OpenCV VideoCapture (0 for webcam, or use a video file path)
cap = cv2.VideoCapture(1)

# Variables to track time and positions for different zones
zones = {
    'sacrum': None,
    'left_shoulder': None,
    'right_shoulder': None,
    'left_hip': None,
    'right_hip': None,
    'elbows': None
}

previous_positions = {
    'left_shoulder': None,
    'right_shoulder': None,
    'left_hip': None,
    'right_hip': None,
    'sacrum': None, 
    'elbows': None
}

movement_threshold = 30
last_change_times = {zone: time.time() for zone in zones}
reposition_alert_interval = 20

def get_zone_positions(landmarks):
    positions = {
        'left_shoulder': (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y),
        'right_shoulder': (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y),
        'left_hip': (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y),
        'right_hip': (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y),
        'sacrum': ((landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2,
                       (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2),
        'elbows': [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x],
    }
    return positions


def check_movement(current_position, previous_position, threshold=movement_threshold):
    if previous_position is None:
        return "moving"  # Initial state assumes movement
    # Calculate Euclidean distance
    distance = math.sqrt((current_position[0] - previous_position[0]) ** 2 + (current_position[1] - previous_position[1]) ** 2)
    return "moving" if distance > threshold else "still"

def get_angle(x1, y1, x2, y2):
    """Calculate the angle of the line connecting two points with respect to the horizontal axis."""
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

def get_posture(landmarks):
    # Get the x and y coordinates of key landmarks
    left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
    right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
    left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
    right_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)
    
    # Calculate the horizontal distances between shoulders and hips
    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
    hip_width = abs(left_hip[0] - right_hip[0])
    
    # Calculate the angle of the shoulder line
    shoulder_angle = get_angle(left_shoulder[0], left_shoulder[1], right_shoulder[0], right_shoulder[1])

    # Determine posture based on distances and angle
    if shoulder_width > 0.15 and hip_width > 0.15:  # Both shoulders and hips are widely apart (lying on back or front)
        return "Straight"
    elif shoulder_width < 0.1 and hip_width < 0.1:  # Shoulders and hips are closer together (lying on side)
        return "Side"
    else:
        return "Unknown"

def has_moved(previous, current, threshold=0.05):
    # Calculate movement distance for multi-point zones (e.g., sacrum, shoulders)
    movement_distance = sum([(current[i] - previous[i]) ** 2 for i in range(len(current))]) ** 0.5
    return movement_distance > threshold

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        
        # Determine patient's posture
        posture = get_posture(results.pose_landmarks.landmark)

        # Display detected posture on the video feed
        cv2.putText(img, f"Posture: {posture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


        # Draw pose landmarks
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get current zone positions
        current_positions = get_zone_positions(results.pose_landmarks.landmark)
        movement_statuses = {}
        key_parts = get_zone_positions(landmarks=results.pose_landmarks.landmark)

        for part, current_position in key_parts.items():
            movement_statuses[part] = check_movement(current_position, previous_positions[part])
            previous_positions[part] = current_position  # Update the previous position

        # Draw boxes and labels for each key part
        for part, position in key_parts.items():
            # Convert normalized coordinates to pixel coordinates
            h, w, _ = img.shape
            x, y = int(position[0] * w), int(position[1] * h)
            # Draw a rectangle around the key part
            cv2.rectangle(img, (x - 20, y - 20), (x + 20, y + 20), (0, 255, 0), 2)
            # Put movement status text near the box
            cv2.putText(img, f"{part}: {movement_statuses[part]}", (x - 40, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)


        # Check movements for each zone
        for zone, last_position in zones.items():
            if last_position:
                # Determine if the zone has moved sufficiently
                if has_moved(last_position, current_positions[zone]):
                    last_change_times[zone] = time.time()  # Reset timer for this zone
                    print(f"{zone.capitalize()} moved; timer reset.")
            
            # Update last known position for the zone
            zones[zone] = current_positions[zone]

        # Check if all zones have been inactive beyond the alert interval
        if all(time.time() - last_change_times[zone] > reposition_alert_interval for zone in zones):
            print("Reposition alert: It's time to change the patient's position!")

    # Display the image
    cv2.imshow("Patient Monitoring - Multi-Zone", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
