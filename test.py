
"""
Real-Time Sign Language Detection
CNN + MediaPipe + OpenCV

Features:
1. Detects all 7 trained gestures.
2. Displays detected gesture names.
3. Shows prediction confidence.
4. Displays top 3 predictions.
5. Uses stability filtering.
6. No Text-to-Speech.
7. Press Q to exit.
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf


# ==================================================
# 1. CONFIGURATION
# ==================================================

MODEL_PATH = "sign_model.h5"
LABEL_PATH = "class_labels.npy"

IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.50
STABLE_FRAMES = 3


# ==================================================
# 2. LOAD MODEL AND LABELS
# ==================================================

print("Loading CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

labels = np.load(
    LABEL_PATH,
    allow_pickle=True
).item()

print("Model loaded successfully!")

print("\nAvailable gestures:")

for index in sorted(labels.keys(), key=lambda x: int(x)):
    print(f"{index}: {labels[index]}")


# ==================================================
# 3. MEDIAPIPE INITIALIZATION
# ==================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)


# ==================================================
# 4. IMAGE PREPROCESSING
# ==================================================

def preprocess(crop):

    # Convert BGR to RGB
    crop = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    # Resize image
    img = cv2.resize(
        crop,
        (IMG_SIZE, IMG_SIZE)
    )

    # Normalize pixel values
    img = img.astype("float32") / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


# ==================================================
# 5. GET GESTURE LABEL
# ==================================================

def get_label(index):

    raw_label = labels.get(
        index,
        labels.get(str(index), str(index))
    )

    if isinstance(raw_label, bytes):
        raw_label = raw_label.decode("utf-8")

    return str(raw_label).replace("_", " ").strip().title()


# ==================================================
# 6. OPEN WEBCAM
# ==================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Error: Cannot open webcam.")

    hands.close()
    raise SystemExit


print("\n================================")
print(" SIGN LANGUAGE DETECTION STARTED")
print("================================")

print("Show your hand gestures.")
print("Press Q to exit.\n")


# ==================================================
# 7. STABILITY VARIABLES
# ==================================================

last_label = ""
stable_count = 0

display_label = "Show Your Hand"

confidence = 0.0

top_predictions = []


# ==================================================
# 8. MAIN DETECTION LOOP
# ==================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Unable to read webcam.")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result = hands.process(rgb)

        display_label = "No Hand Detected"
        confidence = 0.0
        top_predictions = []

        if result.multi_hand_landmarks:

            h, w = frame.shape[:2]

            lm = result.multi_hand_landmarks[0]

            # ======================================
            # DRAW HAND LANDMARKS
            # ======================================

            mp_drawing.draw_landmarks(
                frame,
                lm,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            # ======================================
            # HAND BOUNDING BOX
            # ======================================

            xs = [
                point.x * w
                for point in lm.landmark
            ]

            ys = [
                point.y * h
                for point in lm.landmark
            ]

            padding = 30

            x1 = max(0, int(min(xs)) - padding)
            y1 = max(0, int(min(ys)) - padding)

            x2 = min(w, int(max(xs)) + padding)
            y2 = min(h, int(max(ys)) + padding)

            crop = frame[y1:y2, x1:x2]

            # ======================================
            # CNN PREDICTION
            # ======================================

            if (
                crop.size > 0
                and (x2 - x1) > 10
                and (y2 - y1) > 10
            ):

                inp = preprocess(crop)

                predictions = model.predict(
                    inp,
                    verbose=0
                )[0]

                idx = int(np.argmax(predictions))

                confidence = float(predictions[idx])

                predicted_label = get_label(idx)

                # ==================================
                # TOP THREE PREDICTIONS
                # ==================================

                top_indices = np.argsort(
                    predictions
                )[-3:][::-1]

                for i in top_indices:

                    name = get_label(int(i))
                    score = float(predictions[i])

                    top_predictions.append(
                        (name, score)
                    )

                # ==================================
                # CONFIDENCE CHECK
                # ==================================

                if confidence >= CONFIDENCE_THRESHOLD:

                    # Stability filter
                    if predicted_label == last_label:

                        stable_count += 1

                    else:

                        last_label = predicted_label
                        stable_count = 1

                    # Stable gesture detected
                    if stable_count >= STABLE_FRAMES:

                        display_label = predicted_label

                        print(
                            f"Detected: {predicted_label} "
                            f"({confidence * 100:.1f}%)"
                        )

                    else:

                        display_label = "Detecting..."

                else:

                    last_label = ""
                    stable_count = 0
                    display_label = "Uncertain"

            else:

                last_label = ""
                stable_count = 0
                display_label = "Uncertain"

            # ======================================
            # DRAW BOUNDING BOX
            # ======================================

            color = (
                (0, 255, 120)
                if display_label not in (
                    "Uncertain",
                    "Detecting..."
                )
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

            # Reset when hand disappears
            last_label = ""
            stable_count = 0

        # ==================================================
        # 9. DISPLAY DETECTED GESTURE
        # ==================================================

        h, w = frame.shape[:2]

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, h - 190),
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

        # Gesture name
        cv2.putText(
            frame,
            "GESTURE: " + display_label,
            (15, h - 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 245, 160),
            2,
            cv2.LINE_AA
        )

        # Confidence
        cv2.putText(
            frame,
            f"Confidence: {confidence * 100:.2f}%",
            (15, h - 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 220, 0),
            2,
            cv2.LINE_AA
        )

        # Top predictions
        for i, (name, score) in enumerate(top_predictions):

            text = f"{name}: {score * 100:.1f}%"

            cv2.putText(
                frame,
                text,
                (15 + i * 210, h - 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # Exit instruction
        cv2.putText(
            frame,
            "Press Q to Exit",
            (15, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        # ==================================================
        # 10. SHOW WEBCAM
        # ==================================================

        cv2.imshow(
            "Sign Language Detection",
            frame
        )

        # ==================================================
        # 11. KEYBOARD CONTROL
        # ==================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):

            print("\nExiting camera...")
            break


# ==================================================
# 12. RELEASE RESOURCES
# ==================================================

finally:

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    print("\nCamera closed successfully.")
    print("Application terminated.")
