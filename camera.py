import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load your custom image
meme_img = cv2.imread("myphoto.jpg")  # change file name here

# Webcam
cap = cv2.VideoCapture(0)

def one_hand_up(landmarks):
    if landmarks is None:
        return False
    
    lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # show if ANY ONE hand is raised
    if lw.y < ls.y or rw.y < rs.y:
        return True
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    show = False

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        if one_hand_up(lm):
            show = True

    if show:
        h, w = frame.shape[:2]
        meme_resized = cv2.resize(meme_img, (w, h))
        frame = meme_resized  # FULL image override
    else:
        cv2.putText(frame, "Raise one hand to show image", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Pose Meme", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
