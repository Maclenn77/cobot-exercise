import cv2
import mediapipe as mp
import numpy as np
from xarm.wrapper import XArmAPI

# Configura el xArm
arm = XArmAPI('192.168.1.205')  # Cambia por la IP del robot
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

# Posición inicial
x_pos = 200
arm.set_position(x=x_pos, y=0, z=200, roll=-180, pitch=0, yaw=0, speed=50, wait=True)

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)

# Mapeo de coordenadas
centro_x, centro_y = 320, 240
escala_y, escala_z = 0.5, 0.5

# Filtro EMA
alpha = 0.3

dy_filtrado = 0
dz_filtrado = 200

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                x1 = int(hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1])
                y1 = int(hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0])

                x2 = int(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                y2 = int(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

                # Conversión a mm
                dy = np.clip((x2 - centro_x) * escala_y, -200, 200)
                dz = np.clip((centro_y - y2) * escala_z, 150, 350)

                # Aplicar filtro EMA
                dy_filtrado = alpha * dy + (1 - alpha) * dy_filtrado
                dz_filtrado = alpha * dz + (1 - alpha) * dz_filtrado

                # Mover el robot con suavizado
                #arm.set_position(x=x_pos, y=dy_filtrado, z=dz_filtrado,
                #                 roll=-180, pitch=0, yaw=0, speed=100, wait=False)

                # Control del gripper
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if dist < 50:
                    arm.set_cgpio_digital(0, 1, delay_sec=0)
                elif dist > 100:
                    arm.set_cgpio_digital(0, 0, delay_sec=0)



        cv2.imshow("Control xArm con Ventosa y Filtro", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
arm.disconnect()
