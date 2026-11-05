import cv2
import mediapipe as mp    #Now I will explain the data collection part of my project.
                         #In this step, I  collected gesture images.
import os

GESTURES= ["hello", "no", "please", "sorry", "thank_you", "yes", "i love you"]
SAVE_DIR="dataset"
IMG_SIZE= 128
NUM_IMAGES= 250          # images to capture per gesture

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def collect_for_gesture(gesture_name):                 
    save_path = os.path.join(SAVE_DIR, gesture_name)     ##used the OS module to manage file paths and folders.
                                                          # helps to create separate folders for each gesture and store images properly.
    os.makedirs(save_path, exist_ok=True)

    cap   = cv2.VideoCapture(0)
    count = 0

    print(f"\n>>> Get ready to show gesture: '{gesture_name}'")
    print("Press SPACE to start capturing, Q to quit early.\n")

    capturing = False

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # Draw hand skeleton
            if result.multi_hand_landmarks:
                for lm in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            # Status overlay
            status = f"Capturing: {count}/{NUM_IMAGES}" if capturing else "Press SPACE to start"
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, status, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                capturing = True
            if key == ord('q'):
                break

            # Save cropped hand region ===The hand region is detected and cropped automatically.
            if capturing and result.multi_hand_landmarks:
                h, w = frame.shape[:2]
                lm   = result.multi_hand_landmarks[0]
                xs   = [p.x * w for p in lm.landmark]
                ys   = [p.y * h for p in lm.landmark]
                pad  = 30
                x1   = max(0, int(min(xs)) - pad)
                y1   = max(0, int(min(ys)) - pad)
                x2   = min(w, int(max(xs)) + pad)
                y2   = min(h, int(max(ys)) + pad)

                crop    = frame[y1:y2, x1:x2]
                resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(os.path.join(save_path, f"{count}.jpg"), resized)   #Each image is saved in a folder for training.
                count += 1

                if count >= NUM_IMAGES:
                    print(f" Done collecting '{gesture_name}'!")
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=== Sign Language Data Collector ===")
    print("Gestures:", GESTURES)
    for g in GESTURES:
        collect_for_gesture(g)
    print("\n All gestures collected! Run 2_train_model.py next.")  #this part is used for hand detection.