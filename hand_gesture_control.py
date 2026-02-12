import os
import sys
import cv2
import mediapipe as mp
# Some opencv builds may not expose __version__; ensure it's present for pyautogui/pyscreeze
if not hasattr(cv2, '__version__'):
    cv2.__version__ = '4.7.0'

import pyautogui
import numpy as np
import time
import time
import argparse
import subprocess
import shutil
# CLI flags
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--no-actions', action='store_true', help='Disable system actions (safe debug mode)')
parser.add_argument('--cooldown', type=float, default=0.8, help='Cooldown (seconds) between repeated actions for same gesture')
parser.add_argument('--focus-window', type=str, default=None, help='Optional window name to focus via xdotool before sending keys')
# Use a minimal parse so downstream code can still use argparse if needed
args, _ = parser.parse_known_args()
NO_ACTIONS = bool(getattr(args, 'no_actions', False))
ACTION_COOLDOWN = float(getattr(args, 'cooldown', 0.8))
FOCUS_WINDOW = getattr(args, 'focus_window', None)

# action debounce state
_last_action_time = 0.0
_last_gesture = None
# Gesture detection and control helpers (defined early so callback can call them)
def get_hand_gesture(landmarks):
    # landmarks: list of NormalizedLandmark-like objects with .x and .y
    landmark_enum = mp_hands.HandLandmark if 'mp_hands' in globals() and mp_hands is not None else None
    if landmark_enum is None:
        try:
            landmark_enum = __import__('mediapipe').tasks.python.vision.hand_landmarker.HandLandmark
        except Exception:
            # fallback numeric indices
            class landmark_enum:  # type: ignore
                WRIST=0; THUMB_TIP=4; INDEX_FINGER_TIP=8; MIDDLE_FINGER_TIP=12; RING_FINGER_TIP=16; PINKY_TIP=20

    # Extract thumb and finger tips
    thumb_tip = landmarks[landmark_enum.THUMB_TIP]
    index_tip = landmarks[landmark_enum.INDEX_FINGER_TIP]
    middle_tip = landmarks[landmark_enum.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[landmark_enum.RING_FINGER_TIP]
    pinky_tip = landmarks[landmark_enum.PINKY_TIP]
    wrist = landmarks[landmark_enum.WRIST]

    # Calculate distances between thumb and index tip, and index and middle tip
    distance_thumb_index = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
    distance_index_middle = np.linalg.norm(np.array([index_tip.x, index_tip.y]) - np.array([middle_tip.x, middle_tip.y]))

    print(f"Thumb-Index Distance: {distance_thumb_index}, Index-Middle Distance: {distance_index_middle}")

    # Finger extended heuristics using distance from wrist
    def is_extended(tip, pip):
        tip_dist = np.linalg.norm(np.array([tip.x - wrist.x, tip.y - wrist.y]))
        pip_dist = np.linalg.norm(np.array([pip.x - wrist.x, pip.y - wrist.y]))
        return tip_dist > pip_dist * 1.02

    # pip/ip joints for each finger
    try:
        index_pip = landmarks[landmark_enum.INDEX_FINGER_PIP]
        middle_pip = landmarks[landmark_enum.MIDDLE_FINGER_PIP]
        ring_pip = landmarks[landmark_enum.RING_FINGER_PIP]
        pinky_pip = landmarks[landmark_enum.PINKY_PIP]
        thumb_ip = landmarks[landmark_enum.THUMB_IP]
    except Exception:
        # Fallback indices if enum isn't available
        index_pip = landmarks[6]
        middle_pip = landmarks[10]
        ring_pip = landmarks[14]
        pinky_pip = landmarks[18]
        thumb_ip = landmarks[3]

    index_ext = is_extended(index_tip, index_pip)
    middle_ext = is_extended(middle_tip, middle_pip)
    ring_ext = is_extended(ring_tip, ring_pip)
    pinky_ext = is_extended(pinky_tip, pinky_pip)
    thumb_ext = is_extended(thumb_tip, thumb_ip)

    # New gestures (check specific cases first)
    # Pinch: thumb and index very close, other fingers folded
    if distance_thumb_index < 0.04 and not (middle_ext or ring_ext or pinky_ext):
        print("Gesture: Pinch detected")
        return "Pinch"

    # OK: thumb and index touching while middle and ring are extended (common OK pose)
    if distance_thumb_index < 0.06 and middle_ext and not ring_ext and not pinky_ext:
        print("Gesture: OK detected")
        return "OK"

    # Peace: index and middle extended, ring and pinky folded
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        print("Gesture: Peace detected")
        return "Peace"

    # Pointing: only the index finger extended
    if index_ext and not (middle_ext or ring_ext or pinky_ext):
        print("Gesture: Pointing detected")
        return "Pointing"

    # Fist: all tips close to each other (approx)
    if distance_thumb_index < 0.1 and distance_index_middle < 0.06:
        print("Gesture: Fist detected")
        return "Fist"

    # Open hand: wide index-middle spacing
    if distance_index_middle > 0.15:
        print("Gesture: Open Hand detected")
        return "Open Hand"

    # Thumb up / down (vertical relative to wrist)
    if thumb_tip.y < wrist.y and abs(thumb_tip.x - index_tip.x) < 0.1:
        print("Gesture: Thumb Up detected")
        return "Thumb Up"
    if thumb_tip.y > wrist.y and abs(thumb_tip.x - index_tip.x) < 0.1:
        print("Gesture: Thumb Down detected")
        return "Thumb Down"

    print("Gesture: Unknown")
    return "Unknown"

def control_application(gesture):
    print(f"Action for Gesture: {gesture}")
    if NO_ACTIONS:
        print("Actions are disabled (no-actions).")
        return
    global _last_action_time, _last_gesture
    now = time.time()
    if gesture == _last_gesture and (now - _last_action_time) < ACTION_COOLDOWN:
        print(f"Suppressed repeated action for '{gesture}' (cooldown)")
        return

    # Prefer system command `pactl` (PulseAudio / PipeWire) for reliable volume control on Linux.
    pactl_path = shutil.which('pactl')
    amixer_path = shutil.which('amixer')
    xdotool_path = shutil.which('xdotool')

    def _maybe_focus():
        if FOCUS_WINDOW and xdotool_path:
            try:
                # find window ids by name, activate the first
                result = subprocess.run(['xdotool', 'search', '--name', FOCUS_WINDOW], capture_output=True, text=True)
                winids = [w for w in result.stdout.strip().splitlines() if w.strip()]
                if winids:
                    winid = winids[0]
                    subprocess.run(['xdotool', 'windowactivate', '--sync', winid])
                    time.sleep(0.08)
                    return winid
            except Exception as e:
                print('xdotool focus failed:', e)
        return None

    def _send_key_to_window(key: str, winid: str | None):
        # If we have an explicit window id and xdotool, send the key to that window
        if winid and xdotool_path:
            try:
                # First ensure the window is focused
                subprocess.run(['xdotool', 'windowfocus', '--sync', winid], check=False)
                time.sleep(0.05)
                # Then send the key to the focused window
                subprocess.run(['xdotool', 'key', key], check=False)
                time.sleep(0.08)
                print(f"Sent key '{key}' to window {winid}")
                return True
            except Exception as e:
                print(f'xdotool key failed: {e}', file=sys.stderr)
                # fall through to pyautogui
        try:
            pyautogui.press(key.lower() if len(key) > 1 else key)
            time.sleep(0.06)
            print(f"Sent key '{key}' via pyautogui")
            return True
        except Exception as e:
            print(f'pyautogui press failed: {e}', file=sys.stderr)
            return False

    def _pactl_change(delta: str):
        try:
            subprocess.run(['pactl', 'set-sink-volume', '@DEFAULT_SINK@', delta], check=False)
            return True
        except Exception as e:
            print('pactl failed:', e)
            return False

    def _amixer_change(delta: str):
        try:
            # Use ALSA mixer via pulse plugin; delta should be like '+5%' or '-5%'
            if delta.startswith('+'):
                amt = delta[1:]
                subprocess.run(['amixer', '-D', 'pulse', 'sset', 'Master', f'{amt}%+'], check=False)
            else:
                amt = delta[1:] if delta.startswith('-') else delta
                subprocess.run(['amixer', '-D', 'pulse', 'sset', 'Master', f'{amt}%-'], check=False)
            return True
        except Exception as e:
            print('amixer failed:', e)
            return False

    # focus window if requested
    winid = _maybe_focus()
    # If no explicit window focus requested, auto-search for PowerPoint (LibreOffice/GNOME Impress or MS Office)
    if not winid and xdotool_path:
        try:
            # Try to find a pptx/presentation window with various patterns
            search_patterns = [
                'LibreOffice Impress',
                'Impress', 
                'PowerPoint',
                'Microsoft PowerPoint',
                'Presentation',
                '.pptx',
                '.odp',
            ]
            for pattern in search_patterns:
                print(f"Searching for window matching: '{pattern}'")
                result = subprocess.run(['xdotool', 'search', '--name', pattern], capture_output=True, text=True)
                winids = [w for w in result.stdout.strip().splitlines() if w.strip()]
                if winids:
                    winid = winids[0]
                    print(f"Found presentation window: {pattern} (id={winid})")
                    # Try to activate/focus it
                    subprocess.run(['xdotool', 'windowactivate', '--sync', winid], check=False)
                    time.sleep(0.1)
                    break
            if not winid:
                print("No presentation window found. Will use default focus.")
        except Exception as e:
            print(f"Window search failed: {e}", file=sys.stderr)
            pass

    if gesture == "Open Hand":
        if pactl_path:
            _pactl_change('+5%')
            print("Volume Increased (pactl)")
        elif amixer_path:
            _amixer_change('+5%')
            print("Volume Increased (amixer)")
        else:
            pyautogui.press('volumeup')
            print("Volume Increased (pyautogui)")
    elif gesture == "Fist":
        if pactl_path:
            _pactl_change('-5%')
            print("Volume Decreased (pactl)")
        elif amixer_path:
            _amixer_change('-5%')
            print("Volume Decreased (amixer)")
        else:
            pyautogui.press('volumedown')
            print("Volume Decreased (pyautogui)")
    elif gesture == "Pointing":
        _send_key_to_window('Left', winid)
        print("Previous Slide (Pointing)")
    elif gesture == "Peace":
        _send_key_to_window('Right', winid)
        print("Next Slide (Peace)")
    elif gesture == "OK":
        _send_key_to_window('space', winid)
        print("Play/Pause (OK)")
    elif gesture == "Pinch":
        _send_key_to_window('f', winid)
        print("Toggle Fullscreen (Pinch)")
    elif gesture == "Thumb Up":
        # Mute volume
        if pactl_path:
            _pactl_change('0%')
            print("Volume Muted (pactl)")
        elif amixer_path:
            _amixer_change('0%')
            print("Volume Muted (amixer)")
        else:
            pyautogui.press('volumeoff')
            print("Volume Muted (pyautogui)")
    elif gesture == "Thumb Down":
        # Unmute volume
        if pactl_path:
            _pactl_change('+50%')
            print("Volume Unmuted (pactl)")
        elif amixer_path:
            _amixer_change('+50%')
            print("Volume Unmuted (amixer)")
        else:
            pyautogui.press('volumeup')
            print("Volume Unmuted (pyautogui)")
    else:
        print("No action for gesture:", gesture)
    # record last action
    _last_action_time = time.time()
    _last_gesture = gesture

# to the Tasks API (newer MediaPipe 0.10+). The Tasks API requires a separate
# model file (hand_landmarker.task) which must be downloaded and provided.
USE_SOLUTIONS = False
try:
    mp_solutions = mp.solutions
    mp_hands = mp_solutions.hands
    mp_drawing = mp_solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    USE_SOLUTIONS = True
except Exception:
    # Tasks API imports (MediaPipe 0.10+)
    try:
        from mediapipe.tasks.python.vision import hand_landmarker as hl
        from mediapipe.tasks.python.core import base_options as base_options_lib
        from mediapipe.tasks.python.vision.core import vision_task_running_mode as vtr
        from mediapipe.tasks.python.vision.core import image as mp_image_lib
    except Exception as e:
        print('No usable MediaPipe API found in this environment:', e)
        sys.exit(1)

    # Live-stream hand landmarker requires a model file. Place it in the repo
    # as 'hand_landmarker.task' or update the path below.
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
    if not os.path.exists(MODEL_PATH):
        print('hand_landmarker.task model not found.')
        print('Download it from the MediaPipe repository and place it at:', MODEL_PATH)
        print('Example URL: https://github.com/google/mediapipe/blob/master/mediapipe/tasks/hand_landmarker/hand_landmarker.task')
        sys.exit(1)

    # Callback for live-stream results (will draw on the current frame)
    frame = None
    frame_size = (0, 0)  # (width, height)

    def _on_result(result, image, timestamp_ms):
        global frame, frame_size
        try:
            print(f"_on_result invoked; hands: {len(result.hand_landmarks) if result.hand_landmarks else 0}")
        except Exception:
            print("_on_result invoked; could not read hand_landmarks length")

        if result.hand_landmarks:
            w, h = frame_size
            for hand_landmarks in result.hand_landmarks:
                # Draw connections
                for conn in hl.HandLandmarksConnections.HAND_CONNECTIONS:
                    start = hand_landmarks[conn.start]
                    end = hand_landmarks[conn.end]
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw landmarks
                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                # Gesture detection and control
                try:
                    gesture = get_hand_gesture(hand_landmarks)
                    print(f"Detected Gesture: {gesture}")
                    # Print before taking action
                    control_application(gesture)
                except Exception as e:
                    import traceback
                    print('Error in gesture detection/control:', e)
                    traceback.print_exc()

    base_options = base_options_lib.BaseOptions(model_asset_path=MODEL_PATH)
    options = hl.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vtr.VisionTaskRunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=_on_result,
    )

    landmarker = hl.HandLandmarker.create_from_options(options)


def find_camera(max_index: int = 5):
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        # small delay to allow device probing
        time.sleep(0.1)
        if cap.isOpened():
            print(f'Opened camera at index {i}')
            return cap
        cap.release()
    return None


# Try to locate a usable camera
cap = find_camera(4)
if cap is None:
    print('Could not open any camera indices 0-4.')
    print('Run `ls -l /dev/video*` and `v4l2-ctl --list-devices` to inspect devices.')
    print('You may need to add your user to the `video` group: `sudo usermod -a -G video $USER` then re-login.')
    sys.exit(1)

def get_landmark_enum():
    return mp_hands.HandLandmark if USE_SOLUTIONS else hl.HandLandmark

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB as MediaPipe works on RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if USE_SOLUTIONS:
        # Classic solutions API
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the hand
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                # Get the hand gesture and print it
                gesture = get_hand_gesture(landmarks.landmark)
                print(f"Detected Gesture: {gesture}")  # Debug print
                control_application(gesture)
    else:
        # Tasks API: use async live-stream invocation (we created landmarker in LIVE_STREAM mode)
        mp_image = mp_image_lib.Image(image_format=mp_image_lib.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(time.time() * 1000)

        # set current frame for the callback to draw on
        frame = frame  # ensure local reference
        frame_size = (frame.shape[1], frame.shape[0])
        # Submit async request; results will be handled in _on_result
        landmarker.detect_async(mp_image, timestamp)

    # Show the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
