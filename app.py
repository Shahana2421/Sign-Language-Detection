
"""
Sign Language Detection
CNN + MediaPipe + OpenCV

Features:
1. Detects all 7 trained gestures.
2. Webcam displays only Hello, No, or Uncertain.
3. Detection history stores all 7 actual gestures.
4. No Text-to-Speech.
5. Press Q to exit.
6. Press R to clear history.
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
from collections import deque


# =====================================================
# 1. CONFIGURATION
# =====================================================

MODEL_PATH = "sign_model.h5"
LABEL_PATH = "class_labels.npy"

IMG_SIZE = 128

CONFIDENCE_THRESHOLD = 0.90
STABLE_FRAMES = 3

# Only these two gestures appear by name on camera
VALID_GESTURES = {"Hello", "No"}

# Store the last 5 actual detections
history = deque(maxlen=5)


# =====================================================
# 2. LOAD CNN MODEL AND LABELS
# =====================================================

print("Loading CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

labels = np.load(
    LABEL_PATH,
    allow_pickle=True
).item()

print("Model loaded successfully!")

print("Available gestures:")

for i in range(len(labels)):
    print(i, ":", labels.get(i, labels.get(str(i), "Unknown")))


# =====================================================
# 3. MEDIAPIPE INITIALIZATION
# =====================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.60
)


# =====================================================
# 4. IMAGE PREPROCESSING
# =====================================================

def preprocess(crop):

    # OpenCV uses BGR; training images are loaded as RGB
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    return img


# =====================================================
# 5. GET CLASS LABEL
# =====================================================

def get_label(idx):

    if isinstance(labels, dict):

        raw_label = labels.get(
            idx,
            labels.get(str(idx), str(idx))
        )

    else:

        raw_label = labels[idx]

    if isinstance(raw_label, bytes):
        raw_label = raw_label.decode("utf-8")

    return str(raw_label).replace("_", " ").strip().title()


# =====================================================
# 6. WEBCAM INITIALIZATION
# =====================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Error: Cannot open webcam.")

    hands.close()
    raise SystemExit

print("\nCamera started successfully!")
print("Press Q to exit.")
print("Press R to clear history.")


# =====================================================
# 7. STABILITY VARIABLES
# =====================================================

last_label = ""
stable_count = 0

last_recorded_label = ""

display_label = "Show Hand"
confidence = 0.0


# =====================================================
# 8. MAIN CAMERA LOOP
# =====================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Unable to read camera frame.")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result = hands.process(rgb)

        confidence = 0.0

        if result.multi_hand_landmarks:

            h, w = frame.shape[:2]

            lm = result.multi_hand_landmarks[0]

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                lm,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            # Calculate hand bounding box
            xs = [
                int(point.x * w)
                for point in lm.landmark
            ]

            ys = [
                int(point.y * h)
                for point in lm.landmark
            ]

            padding = 30

            x1 = max(0, min(xs) - padding)
            y1 = max(0, min(ys) - padding)

            x2 = min(w, max(xs) + padding)
            y2 = min(h, max(ys) + padding)

            crop = frame[y1:y2, x1:x2]

            display_label = "Uncertain"

            if crop.size > 0 and (x2 - x1) > 10 and (y2 - y1) > 10:

                # Preprocess hand image
                inp = preprocess(crop)

                # CNN prediction
                predictions = model.predict(
                    inp,
                    verbose=0
                )[0]

                idx = int(np.argmax(predictions))

                confidence = float(predictions[idx])

                predicted_label = get_label(idx)

                # =====================================
                # CONFIDENCE CHECK
                # =====================================

                if confidence >= CONFIDENCE_THRESHOLD:

                    # Stability filter
                    if predicted_label == last_label:

                        stable_count += 1

                    else:

                        last_label = predicted_label
                        stable_count = 1

                    if stable_count >= STABLE_FRAMES:

                        # =================================
                        # CAMERA DISPLAY FILTER
                        # =================================

                        if predicted_label in VALID_GESTURES:

                            display_label = predicted_label

                        else:

                            display_label = "Uncertain"

                        # =================================
                        # SAVE ALL 7 GESTURES TO HISTORY
                        # =================================

                        if predicted_label != last_recorded_label:

                            history.appendleft({
                                "word": predicted_label,
                                "confidence": confidence,
                                "time": time.strftime("%H:%M:%S")
                            })

                            last_recorded_label = predicted_label

                            print(
                                f"Detected: {predicted_label} "
                                f"({confidence * 100:.1f}%)"
                            )

                else:

                    last_label = ""
                    stable_count = 0
                    display_label = "Uncertain"

            # Draw bounding box
            color = (
                (0, 255, 120)
                if display_label in VALID_GESTURES
                else (0, 140, 255)
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

        else:

            # No hand detected
            last_label = ""
            stable_count = 0
            last_recorded_label = ""

            display_label = "No Hand Detected"
            confidence = 0.0

        # =================================================
        # 9. CAMERA DISPLAY
        # =================================================

        h, w = frame.shape[:2]

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, h - 125),
            (w, h),
            (15, 17, 26),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.80,
            frame,
            0.20,
            0,
            frame
        )

        # Display gesture
        cv2.putText(
            frame,
            "GESTURE: " + display_label,
            (15, h - 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 245, 160),
            2,
            cv2.LINE_AA
        )

        # Display confidence
        if confidence > 0:

            cv2.putText(
                frame,
                f"Confidence: {confidence * 100:.1f}%",
                (15, h - 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 220, 0),
                2,
                cv2.LINE_AA
            )

        # Keyboard instructions
        cv2.putText(
            frame,
            "Q: Exit  |  R: Clear History",
            (15, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        # =================================================
        # 10. DETECTION HISTORY
        # =================================================

        panel_width = 280
        x_start = max(10, w - panel_width)

        cv2.rectangle(
            frame,
            (x_start, 40),
            (w - 10, 250),
            (25, 28, 38),
            -1
        )

        cv2.putText(
            frame,
            "DETECTION HISTORY",
            (x_start + 10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 245, 160),
            1
        )

        for i, item in enumerate(list(history)[:5]):

            y = 105 + i * 28

            history_text = (
                f"{item['word']} "
                f"{item['confidence'] * 100:.0f}%"
            )

            cv2.putText(
                frame,
                history_text,
                (x_start + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

        # =================================================
        # 11. SHOW CAMERA
        # =================================================

        cv2.imshow(
            "Sign Language Detection",
            frame
        )

        # =================================================
        # 12. KEYBOARD CONTROLS
        # =================================================

        key = cv2.waitKey(1) & 0xFF

        # Press Q to exit
        if key == ord("q") or key == ord("Q"):

            print("Exiting camera...")
            break

        # Press R to clear history
        elif key == ord("r") or key == ord("R"):

            history.clear()
            last_recorded_label = ""

            print("Detection history cleared.")


# =====================================================
# 13. RELEASE RESOURCES
# =====================================================

finally:

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    print("Camera closed successfully.")
