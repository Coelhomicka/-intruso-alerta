#!/usr/bin/env python3
import cv2
import os
import argparse

def collect_faces(person_id: int, save_dir: str, num_samples: int = 50):
    save_path = os.path.join(save_dir, str(person_id))
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (200, 200))
            cv2.imwrite(f"{save_path}/img_{count:03d}.jpg", face_resized)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow("Coleta de amostras", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= num_samples:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Coletadas {count} imagens em {save_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser("collect_faces")
    p.add_argument("--person-id", type=int, required=True)
    p.add_argument("--save-dir", default="dataset")
    p.add_argument("--num-samples", type=int, default=50)
    args = p.parse_args()
    collect_faces(args.person_id, args.save_dir, args.num_samples)
