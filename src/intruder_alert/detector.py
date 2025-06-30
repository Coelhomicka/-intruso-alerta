import cv2
import time
from pathlib import Path

class FaceDetector:
    def __init__(self, trainer_path: Path, lock_threshold: int):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(str(trainer_path))
        self.lock_threshold = lock_threshold
        self.unknown_start = None
        self.photo_taken = False

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        return faces, gray

    def handle_faces(self, frame, faces, gray, save_dir: Path):
        unknown = False
        for (x, y, w, h) in faces:
            roi = cv2.resize(gray[y:y+h, x:x+w], (200,200))
            id_pred, conf = self.recognizer.predict(roi)
            if conf < 50:
                #name, color = f"User {id_pred}", (0,255,0)
                name, color = f"Mickael", (0,255,0)
            else:
                name, color = "Desconhecido", (0,0,255)
                unknown = True

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                frame,
                f"{name} ({conf:.0f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        # lógica de timer e foto
        if unknown:
            if self.unknown_start is None:
                self.unknown_start = time.time()
            else:
                elapsed = time.time() - self.unknown_start
                if elapsed >= self.lock_threshold and not self.photo_taken:
                    ts = int(time.time())
                    fname = save_dir / f"intruder_{ts}.jpg"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(fname), frame)
                    self.photo_taken = True
                    return True, fname  # sinaliza para notificar e bloquear
        else:
            # reset se voltou usuário conhecido
            if len(faces) > 0:
                self.unknown_start = None
                self.photo_taken = False

        return False, None
