from ultralytics import YOLO
from flask import Flask, Response, render_template, jsonify, request, session, redirect
import cv2, os, threading, time
from datetime import datetime

# ================= CONFIG =================
WEAPON_MODEL_PATH = "model/weapon_detector.pt"
COCO_MODEL_PATH = "yolov8n.pt"
CAM_INDEX = 0

CONF_THRESHOLD = 0.5
IMG_SIZE = 416

SNAPSHOT_DIR = "static/snapshots"

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"
SECRET_KEY = "dark_blue_secret"

# ================= APP =================
app = Flask(__name__)
app.secret_key = SECRET_KEY

weapon_model = YOLO(WEAPON_MODEL_PATH)
general_model = YOLO(COCO_MODEL_PATH)

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ================= LIVE STATE =================
latest_frame = None
frame_lock = threading.Lock()
detection_active = True
last_snapshot_time = 0

# ✅ NORMAL OBJECT MEMORY LOG
NORMAL_DETECTIONS = []

# ================= YOLO THREAD =================
def yolo_worker():
    global latest_frame, last_snapshot_time

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        annotated = frame.copy()

        # ---- WEAPON DETECTION (SNAPSHOT) ----
        results = weapon_model(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
        for r in results:
            if not r.boxes:
                continue

            for box in r.boxes:
                label = r.names[int(box.cls)].upper()
                conf = float(box.conf)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(
                    annotated,
                    f"{label} {conf:.2f}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,255),
                    2
                )

                now = time.time()
                if now - last_snapshot_time > 2:
                    last_snapshot_time = now
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(
                        os.path.join(SNAPSHOT_DIR, f"{label}_{ts}.jpg"),
                        annotated,
                        [cv2.IMWRITE_JPEG_QUALITY, 95]
                    )

        # ---- NORMAL OBJECTS (TEXT ONLY) ----
        results = general_model(frame, conf=0.4, imgsz=IMG_SIZE, verbose=False)
        for r in results:
            if not r.boxes:
                continue
            for box in r.boxes:
                label = r.names[int(box.cls)]
                NORMAL_DETECTIONS.append({
                    "label": label,
                    "time": datetime.now().strftime("%H:%M:%S")
                })

                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),1)
                cv2.putText(
                    annotated,
                    label,
                    (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    1
                )

        with frame_lock:
            latest_frame = annotated.copy()

# ================= STREAM =================
def gen_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

# ================= AUTH =================
@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == ADMIN_USERNAME and request.form["password"] == ADMIN_PASSWORD:
            session["logged_in"] = True
            return redirect("/home")
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ================= PAGES =================
@app.route("/home")
def home():
    if not session.get("logged_in"):
        return redirect("/")
    return render_template("index.html")

@app.route("/detections")
def detections():
    if not session.get("logged_in"):
        return redirect("/")
    return render_template("detections.html")

# ================= API =================
@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/snapshots")
def snapshots():
    data = []
    for f in sorted(os.listdir(SNAPSHOT_DIR), reverse=True):
        if not f.lower().endswith(".jpg"):
            continue
        name = f[:-4]
        parts = name.split("_")
        if len(parts) < 3:
            continue
        data.append({
            "label": "_".join(parts[:-2]),
            "file": f,
            "time": parts[-2] + "_" + parts[-1]
        })
    return jsonify(data)

@app.route("/api/normal_detections")
def normal_detections():
    return jsonify(NORMAL_DETECTIONS[-50:])
@app.route("/settings")
def settings():
    if not session.get("logged_in"):
        return redirect("/")
    return render_template("settings.html")

# ================= RUN =================
if __name__ == "__main__":
    threading.Thread(target=yolo_worker, daemon=True).start()
    app.run(debug=False, threaded=True)
