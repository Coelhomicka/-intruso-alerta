#!/usr/bin/env python3
import cv2
import os
import numpy as np
import argparse

def load_dataset(path: str):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face_samples, ids = [], []

    for person_folder in os.listdir(path):
        person_path = os.path.join(path, person_folder)
        if not os.path.isdir(person_path):
            continue
        label = int(person_folder)
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (200, 200))
            faces = face_cascade.detectMultiScale(img)
            for (x, y, w, h) in faces:
                face_samples.append(img[y:y+h, x:x+w])
                ids.append(label)

    return face_samples, ids

def train_lpbph(dataset_path: str, output_path: str):
    faces, labels = load_dataset(dataset_path)
    print(f"Total de faces: {len(faces)}, Labels: {len(labels)}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.write(output_path)
    print(f"Treinamento concluído e modelo salvo em {output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser("train_model")
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--output", default="trainer.yml")
    args = p.parse_args()
    train_lpbph(args.dataset, args.output)
