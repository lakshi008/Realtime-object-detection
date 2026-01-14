from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import messagebox, ttk
import sqlite3
import os
from datetime import datetime, timedelta
from playsound import playsound
import threading

# ======================
# CONFIG
# ======================

MODEL_PATH = r"C:\Users\Lakshitha P\Realtime\yolov8n.pt"
model = YOLO(MODEL_PATH)

# Snapshot folder
if not os.path.exists("snapshots"):
    os.makedirs("snapshots")

# ======================
# DATABASES
# ======================

# DB 1 – REGULAR OBJECTS
reg_conn = sqlite3.connect("regular_objects.db")
reg_cur = reg_conn.cursor()
reg_cur.execute("""
CREATE TABLE IF NOT EXISTS regular_objects(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object TEXT,
    time TEXT
)
""")
reg_conn.commit()

# DB 2 – DANGEROUS OBJECTS
danger_conn = sqlite3.connect("dangerous_objects.db")
danger_cur = danger_conn.cursor()
danger_cur.execute("""
CREATE TABLE IF NOT EXISTS dangerous_objects(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object TEXT,
    time TEXT,
    snapshot TEXT
)
""")
danger_conn.commit()

# ======================
# FUNCTIONS
# ======================

def play_alert_sound():
    threading.Thread(target=lambda: playsound(r"C:\Users\Lakshitha P\Downloads\alert-109578.mp3"), daemon=True).start()

def show_popup():
    messagebox.showwarning("⚠️ ALERT", "Dangerous object detected!")

def view_regular_records():
    win = tk.Toplevel(root)
    win.title("Regular Objects Log")
    win.geometry("700x400")

    tree = ttk.Treeview(win, columns=("ID", "Object", "Time"), show="headings")
    tree.heading("ID", text="ID")
    tree.heading("Object", text="Object")
    tree.heading("Time", text="Time")
    tree.pack(fill=tk.BOTH, expand=True)

    reg_cur.execute("SELECT * FROM regular_objects")
    for row in reg_cur.fetchall():
        tree.insert("", tk.END, values=row)

def view_danger_records():
    win = tk.Toplevel(root)
    win.title("Dangerous Objects Log")
    win.geometry("800x400")

    tree = ttk.Treeview(win, columns=("ID", "Object", "Time", "Snapshot"), show="headings")
    tree.heading("ID", text="ID")
    tree.heading("Object", text="Object")
    tree.heading("Time", text="Time")
    tree.heading("Snapshot", text="Snapshot")
    tree.pack(fill=tk.BOTH, expand=True)

    danger_cur.execute("SELECT * FROM dangerous_objects")
    for row in danger_cur.fetchall():
        tree.insert("", tk.END, values=row)


def start_detection():
    cap = cv2.VideoCapture(0)
    snapshot_counter = 0
    last_danger_time = None

    dangerous_labels = ["knife", "gun", "sharp_object"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = datetime.now()

        # Run YOLO detection
        results = model(frame, stream=True)

        for result in results:
            for box in result.boxes:
                cls = int(box.cls)
                label = result.names[cls]
                detected_class = label.lower()

                # Coordinates
                xyxy = box.xyxy[0]
                x1, y1, x2, y2 = map(int, xyxy)
                confidence = float(box.conf)

                # Draw Rectangle
                color = (0, 0, 255) if detected_class in dangerous_labels else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

                # If regular object → store in regular_objects DB
                if detected_class not in dangerous_labels:
                    reg_cur.execute(
                        "INSERT INTO regular_objects(object, time) VALUES(?, ?)",
                        (label, now.strftime("%Y-%m-%d %H:%M:%S"))
                    )
                    reg_conn.commit()

                # If dangerous object detected
                else:
                    # Trigger alert only on NEW detection
                    if last_danger_time is None or (now - last_danger_time).seconds > 3:
                        play_alert_sound()
                        show_popup()
                        last_danger_time = now

                    # Save snapshot for 1 minute after detection
                    if last_danger_time is not None and (now - last_danger_time) < timedelta(minutes=1):
                        snap_path = f"snapshots/danger_{snapshot_counter}.jpg"
                        cv2.imwrite(snap_path, frame)
                        snapshot_counter += 1

                        danger_cur.execute(
                            "INSERT INTO dangerous_objects(object, time, snapshot) VALUES(?, ?, ?)",
                            (label, now.strftime("%Y-%m-%d %H:%M:%S"), snap_path)
                        )
                        danger_conn.commit()

        # Show camera
        cv2.imshow("Security System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ======================
# UI SETUP
# ======================

root = tk.Tk()
root.title("Smart Security System")
root.geometry("400x350")
root.configure(bg="#222")

title = tk.Label(root, text="Smart Security System", fg="white", bg="#222",
                 font=("Arial", 18, "bold"))
title.pack(pady=20)

btn1 = tk.Button(root, text="Start Detection", bg="#0078D7", fg="white",
                 font=("Arial", 13), width=22, command=start_detection)
btn1.pack(pady=10)

btn2 = tk.Button(root, text="View Regular Objects Log", bg="#28A745", fg="white",
                 font=("Arial", 13), width=22, command=view_regular_records)
btn2.pack(pady=10)

btn3 = tk.Button(root, text="View Dangerous Objects Log", bg="#DC3545", fg="white",
                 font=("Arial", 13), width=22, command=view_danger_records)
btn3.pack(pady=10)

btn4 = tk.Button(root, text="Exit", bg="gray", fg="white",
                 font=("Arial", 13), width=22, command=root.destroy)
btn4.pack(pady=15)
root.mainloop()