"""
Control an xArm with one hand tracked by MediaPipe.

- Wrist position (X/Y in the camera frame) drives the arm's Y/Z position in
  a fixed vertical plane (arm depth/X stays constant). The wrist is used
  instead of the index fingertip so that pinching to close the gripper
  doesn't also drag the arm's tracked position.
- Wrist rotation (the twist of the wrist->middle-finger-MCP vector, in the
  camera plane) drives the end effector's yaw, so rotating your wrist
  rotates the gripper.
- Pinch distance between thumb tip and index fingertip toggles a vacuum
  gripper connected to digital IO 0 (closed/suction-on when pinched).
- The forward/back arm depth (X) is not hand-tracked; it's nudged a fixed
  step at a time with the Up/Down arrow keys or W/S (both do the same
  thing, so either a left or right hand can rest on the keyboard) while
  the video window has focus.
"""
import cv2
import mediapipe as mp
import numpy as np
from xarm.wrapper import XArmAPI

# ---- Configuration ----
ROBOT_IP = '192.168.1.205'   # xArm IP address
CAMERA_INDEX = 1             # OpenCV camera index

X_HOME = 200                 # Starting forward/back distance (mm)
X_MIN, X_MAX = 150, 350      # Forward/back travel bounds (mm)
X_STEP = 5                   # mm nudged per Up/Down or W/S key press
Y_LIMIT = 200                # Max +/- Y travel from center (mm)
Z_MIN, Z_MAX = 150, 350      # Vertical travel bounds (mm)
Z_HOME = (Z_MIN + Z_MAX) / 2

# Arrow-key codes returned by cv2.waitKeyEx() vary by platform/backend, so
# cover the common ones (Windows, Linux/GTK, macOS/Cocoa).
KEY_UP = {2490368, 65362, 63232}
KEY_DOWN = {2621440, 65364, 63233}

SCALE_Y, SCALE_Z = 0.5, 0.5  # pixel-to-mm scale factors
EMA_ALPHA = 0.3               # smoothing factor for exponential moving average

YAW_LIMIT = 90                 # max +/- wrist-twist rotation applied to the gripper (deg)

SPEED = 200                   # mm/s for servo streaming
MVACC = 2000                  # mm/s^2

PINCH_CLOSE_DIST = 50         # px distance below which gripper closes (suction on)
PINCH_OPEN_DIST = 100         # px distance above which gripper opens (suction off)

GRIPPER_IO = 0


def connect_arm():
    arm = XArmAPI(ROBOT_IP)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    if arm.error_code != 0:
        arm.clean_error()
        arm.motion_enable(enable=True)
        arm.set_state(state=0)

    arm.set_cgpio_digital(GRIPPER_IO, 0, delay_sec=0)

    # Move to a safe starting pose before switching to streaming mode.
    arm.set_position(x=X_HOME, y=0, z=Z_HOME, roll=-180, pitch=0, yaw=0,
                      speed=50, wait=True)

    # Servo (streaming) mode: designed for frequent, low-latency position
    # updates, unlike mode 0 which queues each set_position as a discrete move.
    arm.set_mode(1)
    arm.set_state(state=0)
    return arm


def main():
    arm = connect_arm()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    x_pos = X_HOME
    dy_filtered = 0.0
    dz_filtered = Z_HOME
    yaw_filtered = 0.0
    gripper_closed = False

    try:
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                height, width, _ = frame.shape
                center_x, center_y = width / 2, height / 2

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                    x1 = hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width
                    y1 = hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height

                    x2 = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width
                    y2 = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height

                    # Wrist position drives arm Y/Z. Unlike the fingertips, it doesn't
                    # move when pinching, so closing the gripper no longer drags the arm.
                    wx = hand.landmark[mp_hands.HandLandmark.WRIST].x * width
                    wy = hand.landmark[mp_hands.HandLandmark.WRIST].y * height

                    mx = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * width
                    my = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * height

                    # Map wrist position (pixels) to arm Y/Z (mm), clipped to safe bounds.
                    dy = np.clip((wx - center_x) * SCALE_Y, -Y_LIMIT, Y_LIMIT)
                    dz = np.clip((center_y - wy) * SCALE_Z + Z_HOME, Z_MIN, Z_MAX)

                    # Wrist-twist angle (wrist -> middle-finger-MCP vector, in the camera
                    # plane): 0 deg when the hand points straight up toward the camera.
                    yaw = np.clip(np.degrees(np.arctan2(mx - wx, wy - my)),
                                  -YAW_LIMIT, YAW_LIMIT)

                    dy_filtered = EMA_ALPHA * dy + (1 - EMA_ALPHA) * dy_filtered
                    dz_filtered = EMA_ALPHA * dz + (1 - EMA_ALPHA) * dz_filtered
                    yaw_filtered = EMA_ALPHA * yaw + (1 - EMA_ALPHA) * yaw_filtered

                    arm.set_servo_cartesian(
                        [x_pos, dy_filtered, dz_filtered, -180, 0, yaw_filtered],
                        speed=SPEED, mvacc=MVACC)

                    # Pinch gesture (thumb tip <-> index fingertip) controls the vacuum
                    # gripper, with hysteresis so we only send an IO command on state
                    # changes. This no longer affects arm position (see wrist tracking above).
                    dist = np.hypot(x2 - x1, y2 - y1)
                    if dist < PINCH_CLOSE_DIST and not gripper_closed:
                        arm.set_cgpio_digital(GRIPPER_IO, 1, delay_sec=0)
                        gripper_closed = True
                    elif dist > PINCH_OPEN_DIST and gripper_closed:
                        arm.set_cgpio_digital(GRIPPER_IO, 0, delay_sec=0)
                        gripper_closed = False

                    cv2.putText(frame, f"x={x_pos:.0f} y={dy_filtered:.0f} z={dz_filtered:.0f} "
                                        f"yaw={yaw_filtered:.0f} pinch={dist:.0f} "
                                        f"grip={'CLOSED' if gripper_closed else 'OPEN'}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"x={x_pos:.0f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Control xArm con Ventosa y Filtro", frame)

                key = cv2.waitKeyEx(1)
                if key == 27:
                    break
                elif key in KEY_UP or key in (ord('w'), ord('W')):
                    x_pos = min(x_pos + X_STEP, X_MAX)
                elif key in KEY_DOWN or key in (ord('s'), ord('S')):
                    x_pos = max(x_pos - X_STEP, X_MIN)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        arm.set_cgpio_digital(GRIPPER_IO, 0, delay_sec=0)
        arm.set_mode(0)
        arm.set_state(state=0)
        arm.disconnect()


if __name__ == '__main__':
    main()
