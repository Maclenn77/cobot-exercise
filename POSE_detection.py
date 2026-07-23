import cv2
import mediapipe as mp


##Initialize pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
#pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


cap = cv2.VideoCapture(1)

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1) as holistic:
    
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        
        #MANO IZQUIERDA
        mp_drawing.draw_landmarks(
    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=1))
            
        #MANO DERECHA
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(57, 143, 0), thickness=2))
            
        #POSTURA
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
            
        print(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE])
    
        
        cv2.imshow("Frame", frame)
         
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
cap.release()
cv2.destroyAllWindows()
