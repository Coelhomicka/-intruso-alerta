import argparse
import cv2
import logging
from pathlib import Path

from intruder_alert.config import load_config
from intruder_alert.detector import FaceDetector
from intruder_alert.locker import lock_screen

def parse_args():
    p = argparse.ArgumentParser("intruder-alert")
    p.add_argument("--threshold", type=int, help="Segundos até lock")
    p.add_argument("--save-dir", type=Path, help="Onde salvar fotos")
    return p.parse_args()

def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s"
    )
    cfg = load_config()
    args = parse_args()
    lock_threshold = args.threshold or cfg["lock_threshold"]
    save_dir = args.save_dir or cfg["save_dir"]

    detector = FaceDetector(Path(cfg["trainer"]["output"]), lock_threshold)

    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces, gray = detector.process_frame(frame)
            should_alert, photo_path = detector.handle_faces(frame, faces, gray, save_dir)

            cv2.imshow("Intruder Alert", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if should_alert:
                logging.warning(f"Intruder detected! Photo saved at {photo_path}")
                lock_screen()

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
