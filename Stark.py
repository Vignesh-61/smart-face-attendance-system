import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from ultralytics import YOLO
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Label, Toplevel, messagebox, simpledialog
import threading

# Paths setup
KNOWN_FACES_DIR = 'known_faces'
ATTENDANCE_FILE = 'attendance/attendance.csv'

# Load models
face_detector = YOLO('yolov8n-face-lindevs.pt')
weights = ResNet50_Weights.IMAGENET1K_V1
resnet = resnet50(weights=weights)
resnet.fc = torch.nn.Identity()
resnet.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def crop_and_align_face(img, box):
    x1, y1, x2, y2 = map(int, box)
    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, (224, 224))


def get_average_embedding(person_dir):
    embeddings = []
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            emb = resnet(img_tensor).cpu().numpy()
        embeddings.append(emb)
    if embeddings:
        return np.mean(embeddings, axis=0).flatten()
    return None


def load_known_faces():
    known_embeddings = {}
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if os.path.isdir(person_dir):
            embedding = get_average_embedding(person_dir)
            if embedding is not None:
                known_embeddings[person_name] = embedding
    return known_embeddings


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def mark_attendance(name):
    os.makedirs(os.path.dirname(ATTENDANCE_FILE), exist_ok=True)
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
        df.to_csv(ATTENDANCE_FILE, index=False)
    df = pd.read_csv(ATTENDANCE_FILE)
    today = datetime.now().strftime('%Y-%m-%d')
    if not ((df['Name'] == name) & (df['Date'] == today)).any():
        time_now = datetime.now().strftime('%H:%M:%S')
        new_entry = pd.DataFrame([[name, today, time_now]],
                                 columns=['Name', 'Date', 'Time'])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        return "Marked"
    else:
        return "Already Present"


def show_attendance(date=None):
    if not os.path.exists(ATTENDANCE_FILE):
        messagebox.showinfo("Attendance", "No attendance file found.")
        return
    df = pd.read_csv(ATTENDANCE_FILE)
    if date:
        df = df[df['Date'] == date]
    today = datetime.now().strftime('%Y-%m-%d')
    win = Toplevel()
    win.title(f"Attendance{' for ' + date if date else ' for ' + today}")
    if df.empty:
        tk.Label(win, text="No attendance records found!").pack()
    else:
        text = tk.Text(win, width=40, height=15)
        text.insert(tk.END, df.to_string(index=False))
        text.pack()


def add_new_face():
    def capture_images(name, top):
        os.makedirs(os.path.join(KNOWN_FACES_DIR, name), exist_ok=True)

        # Force DirectShow backend on Windows
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                "Cannot open camera (index 0, CAP_DSHOW). "
                "Check permissions or try another index."
            )
            top.destroy()
            return

        count = 0
        messagebox.showinfo(
            "Instructions",
            "Position your face in front of the camera. 5 photos will be taken."
        )

        while count < 5:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Camera Error",
                                     "Failed to read frame from camera.")
                break

            results = face_detector(frame)
            if len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        face_img = frame[y1:y2, x1:x2]
                        if face_img.size == 0:
                            continue
                        img_path = os.path.join(
                            KNOWN_FACES_DIR, name, f"{name}_{count + 1}.jpg"
                        )
                        cv2.imwrite(img_path, face_img)
                        count += 1
                        break

            cv2.imshow('Capture Face - Press Q to abort', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        top.destroy()
        messagebox.showinfo("Done", f"{count} images saved!")

        # Force reload embeddings:
        global known_embeddings
        known_embeddings = load_known_faces()

    def ask_name():
        top = Toplevel()
        top.title("Add New Face")
        tk.Label(top, text="Enter Name:").pack()
        entry = tk.Entry(top)
        entry.pack()

        def ok():
            name = entry.get().strip()
            if not name:
                messagebox.showinfo("Error", "Name cannot be empty!")
                return
            top.withdraw()
            capture_images(name, top)

        tk.Button(top, text="OK", command=ok).pack()

    ask_name()


def start_recognition():
    # Use global so it picks up new faces after registration
    global known_embeddings
    known_embeddings = load_known_faces()

    class RecognitionApp:
        def __init__(self):
            self.rec_window = tk.Toplevel()
            self.rec_window.title("Live Face Recognition")
            self.label_cam = Label(self.rec_window)
            self.label_cam.pack()
            self.label_info = Label(self.rec_window,
                                    text="Starting...",
                                    font=("Helvetica", 16))
            self.label_info.pack()

            # Force DirectShow backend
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                messagebox.showerror(
                    "Camera Error",
                    "Cannot open camera (index 0, CAP_DSHOW)."
                )
                self.rec_window.destroy()
                return

            self.running = True
            threading.Thread(target=self.video_loop, daemon=True).start()
            self.rec_window.protocol("WM_DELETE_WINDOW", self.on_close)

        def video_loop(self):
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    messagebox.showerror("Camera Error",
                                         "Failed to read frame from camera.")
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detector(frame)
                names_detected = []

                if len(results) > 0:
                    result = results[0]
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        for box in boxes:
                            x1, y1, x2, y2 = box
                            face_img = crop_and_align_face(rgb_frame, box)
                            if face_img is None:
                                continue

                            face_pil = Image.fromarray(face_img)
                            face_tensor = preprocess(face_pil).unsqueeze(0)
                            with torch.no_grad():
                                emb = resnet(face_tensor).cpu().numpy().flatten()

                            best_match = None
                            best_score = -1
                            for name, known_emb in known_embeddings.items():
                                score = cosine_similarity(emb, known_emb)
                                if score > best_score:
                                    best_score = score
                                    best_match = name

                            if best_score > 0.87:
                                status = mark_attendance(best_match)
                                display_text = f"{best_match} ({best_score:.2f}) - {status}"
                            else:
                                display_text = "Unknown"

                            names_detected.append(display_text)
                            cv2.rectangle(
                                frame,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (0, 255, 0),
                                2,
                            )
                            cv2.putText(
                                frame,
                                display_text,
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (0, 255, 0),
                                2,
                            )

                self.label_info.config(
                    text=names_detected[-1] if names_detected else "No faces detected"
                )

                img_pil = Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                imgtk = ImageTk.PhotoImage(image=img_pil)
                self.label_cam.imgtk = imgtk
                self.label_cam.configure(image=imgtk)

            if self.cap.isOpened():
                self.cap.release()

        def on_close(self):
            self.running = False
            self.rec_window.destroy()

    RecognitionApp()


class MainMenu:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Recognition Attendance System")
        self.master.geometry("400x350")

        self.btn_add = tk.Button(
            master, text="Add New Face", width=30, height=2,
            command=add_new_face
        )
        self.btn_add.pack(pady=10)

        self.btn_recognize = tk.Button(
            master, text="Start Recognition", width=30, height=2,
            command=start_recognition
        )
        self.btn_recognize.pack(pady=10)

        self.btn_view_today = tk.Button(
            master, text="View Today's Attendance", width=30, height=2,
            command=lambda: show_attendance()
        )
        self.btn_view_today.pack(pady=10)

        self.btn_view_prev = tk.Button(
            master, text="View Previous Date Attendance",
            width=30, height=2,
            command=self.prev_date_dialog
        )
        self.btn_view_prev.pack(pady=10)

        self.btn_exit = tk.Button(
            master, text="Exit", width=30, height=2, command=master.quit
        )
        self.btn_exit.pack(pady=10)

    def prev_date_dialog(self):
        date = simpledialog.askstring(
            "Previous Attendance", "Enter date in YYYY-MM-DD format:"
        )
        if date:
            show_attendance(date)


if __name__ == "__main__":
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
    root = tk.Tk()
    app = MainMenu(root)
    root.mainloop()
