"""
Control an xArm with one hand tracked by MediaPipe.

- Wrist position (X/Y in the camera frame) drives the arm's Y/Z position in
  a fixed vertical plane (arm depth/X stays constant). The wrist is used
  instead of the index fingertip so that pinching to close the gripper
  doesn't also drag the arm's tracked position. A new Y/Z target only takes
  effect once the tracked position has held past the dead-band for
  HOLD_TIME seconds, so brief/involuntary hand movement is ignored and only
  a sustained, deliberate move commits.
- An on-screen box shows the workspace the wrist maps to (matching the
  Y/Z limits below) with a crosshair at the home/center position, so you
  can see where to hold your hand to match a given arm position. The
  tracked-wrist marker is green when settled, yellow while a move is
  "holding" before it commits, and red when the hand has left the box.
- Pinch distance between thumb tip and index fingertip toggles a vacuum
  gripper connected to digital IO 0 (closed/suction-on when pinched).
- The forward/back arm depth (X) is not hand-tracked; it's nudged a fixed
  step at a time with the Up/Down arrow keys or W/S.
- The end effector's yaw is not hand-tracked either (wrist-twist detection
  via MediaPipe proved unreliable in practice); it's nudged with the
  Left/Right arrow keys or A/D instead.
  (X and yaw each respond to both an arrow key and a letter key, so either
  a left or right hand can comfortably rest on the keyboard.)
- Pressing Space appends the arm's current actual pose (x, y, z, roll,
  pitch, yaw) as a row to a CSV file, for building a palletization routine
  from recorded waypoints.
"""
import csv
import os
import time

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

YAW_LIMIT = 90                # Max +/- end-effector yaw (deg)
YAW_STEP = 5                  # deg nudged per Left/Right or A/D key press

# Arrow-key codes returned by cv2.waitKeyEx() vary by platform/backend, so
# cover the common ones (Windows, Linux/GTK, macOS/Cocoa).
KEY_UP = {2490368, 65362, 63232}
KEY_DOWN = {2621440, 65364, 63233}
KEY_LEFT = {2424832, 65361, 63234}
KEY_RIGHT = {2555904, 65363, 63235}
KEY_RECORD = 32   # spacebar; standard ASCII, consistent across platforms

SCALE_Y, SCALE_Z = 0.3, 0.3  # pixel-to-mm scale factors (lower = less arm movement per pixel of hand movement)
EMA_ALPHA = 0.15              # smoothing factor for exponential moving average (lower = smoother/laggier)

# set_servo_cartesian is a streaming interface meant for frequent, small,
# steady updates; our per-camera-frame updates are comparatively sparse and
# noisy, so we run it gently (low speed/accel). A tracked Y/Z target only
# commits once it has stayed past the dead-band continuously for HOLD_TIME
# seconds -- short/unintentional hand movement never reaches the arm at
# all, rather than just being smoothed.
SPEED = 80                   # mm/s for servo streaming
MVACC = 500                  # mm/s^2
POS_DEADBAND = 15            # mm; ignore Y/Z changes smaller than this
HOLD_TIME = 0.4              # seconds a Y/Z change must persist before it commits

PINCH_CLOSE_DIST = 50         # px distance below which gripper closes (suction on)
PINCH_OPEN_DIST = 100         # px distance above which gripper opens (suction off)

GRIPPER_IO = 0

# Anchored to the script's own folder so the CSV always lands next to it,
# regardless of the current working directory the script is launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POINTS_CSV_PATH = os.path.join(SCRIPT_DIR, 'palletization_points.csv')
POINTS_CSV_HEADER = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']


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

    print(f"Recording points to: {POINTS_CSV_PATH}")
    write_header = not os.path.exists(POINTS_CSV_PATH) or os.path.getsize(POINTS_CSV_PATH) == 0
    points_file = open(POINTS_CSV_PATH, 'a', newline='')
    points_writer = csv.writer(points_file)
    if write_header:
        points_writer.writerow(POINTS_CSV_HEADER)
        points_file.flush()
    recorded_count = 0

    limits_text = (f"limits: x[{X_MIN:.0f},{X_MAX:.0f}] y[-{Y_LIMIT:.0f},{Y_LIMIT:.0f}] "
                   f"z[{Z_MIN:.0f},{Z_MAX:.0f}] yaw[-{YAW_LIMIT:.0f},{YAW_LIMIT:.0f}]  "
                   f"(edit these constants at the top of the script)")
    controls_text = "controls: W/S or Up/Down = depth  |  A/D or Left/Right = yaw  |  Space = record point  |  ESC = quit"

    # Pixel-space half-size of the box the wrist maps to, derived from the
    # same scale factors used to convert wrist position into Y/Z (mm).
    half_w_px = Y_LIMIT / SCALE_Y
    half_h_px = ((Z_MAX - Z_MIN) / 2) / SCALE_Z

    x_pos = X_HOME
    yaw_pos = 0
    dy_filtered = 0.0
    dz_filtered = Z_HOME
    last_sent_y, last_sent_z = dy_filtered, dz_filtered
    y_pending_since = None
    z_pending_since = None
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

                # Workspace box + home crosshair: shows where to hold the hand to
                # reach a given arm position, and the safe travel limits at a glance.
                box_tl = (int(center_x - half_w_px), int(center_y - half_h_px))
                box_br = (int(center_x + half_w_px), int(center_y + half_h_px))
                cv2.rectangle(frame, box_tl, box_br, (255, 200, 0), 2)
                cv2.drawMarker(frame, (int(center_x), int(center_y)), (255, 200, 0),
                                markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                now = time.time()

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

                    # Map wrist position (pixels) to arm Y/Z (mm), clipped to safe bounds.
                    dy = np.clip((wx - center_x) * SCALE_Y, -Y_LIMIT, Y_LIMIT)
                    dz = np.clip((center_y - wy) * SCALE_Z + Z_HOME, Z_MIN, Z_MAX)

                    dy_filtered = EMA_ALPHA * dy + (1 - EMA_ALPHA) * dy_filtered
                    dz_filtered = EMA_ALPHA * dz + (1 - EMA_ALPHA) * dz_filtered

                    # A change only commits once it has stayed past the dead-band
                    # continuously for HOLD_TIME seconds; a move that bounces back
                    # within the dead-band before then resets the hold and never
                    # reaches the arm.
                    if abs(dy_filtered - last_sent_y) > POS_DEADBAND:
                        if y_pending_since is None:
                            y_pending_since = now
                        elif now - y_pending_since >= HOLD_TIME:
                            last_sent_y = dy_filtered
                            y_pending_since = None
                    else:
                        y_pending_since = None

                    if abs(dz_filtered - last_sent_z) > POS_DEADBAND:
                        if z_pending_since is None:
                            z_pending_since = now
                        elif now - z_pending_since >= HOLD_TIME:
                            last_sent_z = dz_filtered
                            z_pending_since = None
                    else:
                        z_pending_since = None

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

                    # Wrist marker: red once the hand leaves the mapped box, yellow
                    # while a move is held pending commit, green once settled.
                    if not (box_tl[0] <= wx <= box_br[0] and box_tl[1] <= wy <= box_br[1]):
                        marker_color = (0, 0, 255)
                    elif y_pending_since is not None or z_pending_since is not None:
                        marker_color = (0, 220, 255)
                    else:
                        marker_color = (0, 255, 0)
                    cv2.circle(frame, (int(wx), int(wy)), 10, marker_color, -1)

                    cv2.putText(frame, f"x={x_pos:.0f} y={dy_filtered:.0f} z={dz_filtered:.0f} yaw={yaw_pos:.0f} "
                                        f"pinch={dist:.0f} grip={'CLOSED' if gripper_closed else 'OPEN'}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"x={x_pos:.0f} yaw={yaw_pos:.0f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                arm.set_servo_cartesian(
                    [x_pos, last_sent_y, last_sent_z, -180, 0, yaw_pos],
                    speed=SPEED, mvacc=MVACC)

                cv2.putText(frame, limits_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                cv2.putText(frame, f"points recorded: {recorded_count} (saved to {POINTS_CSV_PATH})",
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                cv2.putText(frame, controls_text, (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

                cv2.imshow("Control xArm con Ventosa y Filtro", frame)

                key = cv2.waitKeyEx(1)
                if key == 27:
                    break
                elif key in KEY_UP or key in (ord('w'), ord('W')):
                    x_pos = min(x_pos + X_STEP, X_MAX)
                elif key in KEY_DOWN or key in (ord('s'), ord('S')):
                    x_pos = max(x_pos - X_STEP, X_MIN)
                elif key in KEY_RIGHT or key in (ord('d'), ord('D')):
                    yaw_pos = min(yaw_pos + YAW_STEP, YAW_LIMIT)
                elif key in KEY_LEFT or key in (ord('a'), ord('A')):
                    yaw_pos = max(yaw_pos - YAW_STEP, -YAW_LIMIT)
                elif key == KEY_RECORD:
                    code, pose = arm.get_position()
                    if code == 0:
                        points_writer.writerow(pose)
                        points_file.flush()
                        recorded_count += 1
                    else:
                        print(f"Could not read arm position (code={code}), point not recorded.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        points_file.close()
        arm.set_cgpio_digital(GRIPPER_IO, 0, delay_sec=0)
        arm.set_mode(0)
        arm.set_state(state=0)
        arm.disconnect()


if __name__ == '__main__':
    main()
