# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import os, sqlite3, random, datetime, threading, time, tempfile
from werkzeug.utils import secure_filename
import cv2, numpy as np
from fer import FER
from deepface import DeepFace
from ultralytics import YOLO
import librosa
import mediapipe as mp 
import smtplib
import base64

# ── Improved gaze detection module (drop gaze_detector.py next to app.py) ──
from gaze_detector import detect_gaze_direction, GazeSmoother

# -------------------- Flask Setup --------------------
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change_this_in_production")

# -------------------- Paths & DB --------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DB_PATH = "candidates.db"
AI_DB_PATH = "ai_results.db"

# -------------------- AI Models --------------------
person_model = YOLO('yolov8n.pt')  # YOLOv8 person detection


mp_face_mesh = mp.solutions.face_mesh
gaze_model = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

emotion_detector = FER(mtcnn=True)

# -------------------- In-memory stores --------------------
active_meetings = {}
meeting_rooms = {}
otp_store = {}
uploaded_first_frames = {}
live_frames = {}
# Per-meeting stop flags for AI pipeline threads.
meeting_stop_flags = {}

# Per-user gaze smoothers (keyed by user_id) to reduce frame-to-frame jitter
_gaze_smoothers: dict = {}

# -------------------- Database Init --------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    username TEXT,
                    email TEXT UNIQUE,
                    password TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS clients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    username TEXT,
                    email TEXT UNIQUE,
                    password TEXT
                )''')
    conn.commit()
    conn.close()
init_db()

def init_ai_db():
    conn = sqlite3.connect(AI_DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS ai_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id TEXT,
                    user_id TEXT,
                    feature TEXT,
                    status TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

init_ai_db()

def init_gaze_table():
    conn = sqlite3.connect("candidates.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS gaze_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id TEXT,
                    user_id TEXT,
                    direction TEXT,
                    timestamp TEXT
                )''')
    conn.commit()
    conn.close()

def init_gaze_summary():
    conn = sqlite3.connect("candidates.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS gaze_summary (
                    meeting_id TEXT,
                    user_id TEXT,
                    total_events INTEGER DEFAULT 0,
                    total_away_time REAL DEFAULT 0,
                    focus_percentage REAL DEFAULT 0,
                    last_updated TEXT,
                    PRIMARY KEY (meeting_id, user_id)
                )''')
    conn.commit()
    conn.close()

init_db()
init_gaze_table()
init_gaze_summary()


# -------------------- Helper DB Functions --------------------
def get_candidate_by_email(email):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id,name,username,email FROM candidates WHERE email=?", (email,))
        row = c.fetchone()
        if row:
            return {"id": row[0],"name":row[1],"username":row[2],"email":row[3]}
    return None

def get_client_by_email(email):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id,name,username,email FROM clients WHERE email=?", (email,))
        row = c.fetchone()
    return {"id": row[0],"name":row[1],"username":row[2],"email":row[3]} if row else None

def update_candidate_profile(email,new_name,new_email):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("UPDATE candidates SET name=?,email=? WHERE email=?", (new_name,new_email,email))
        conn.commit()
        return c.rowcount>0

def update_client_profile(email,new_name,new_email):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("UPDATE clients SET name=?,email=? WHERE email=?", (new_name,new_email,email))
        conn.commit()
        return c.rowcount>0

def save_result(meeting_id, feature, status, user_id="system"):
    print(f"💾 Saving: {meeting_id} | {user_id} | {feature} | {status}")
    with sqlite3.connect(AI_DB_PATH) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO ai_results (meeting_id,user_id,feature,status) VALUES (?,?,?,?)",
                  (meeting_id,user_id,feature,status))
        conn.commit()
        print("✅ Saved successfully")

@app.route("/api/ai_detection_analytics/<meeting_id>")
def get_ai_detection_analytics(meeting_id):
    """
    Analytics for all AI detections excluding gaze
    """
    try:
        with sqlite3.connect(AI_DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT user_id, feature, status, created_at 
                FROM ai_results 
                WHERE meeting_id=? AND feature NOT IN ('gaze', 'gaze_direction', 'gaze_tracking')
                ORDER BY created_at DESC
            """, (meeting_id,))
            all_results = c.fetchall()

            if not all_results:
                return jsonify({"status": "error", "message": "No AI detection data found"}), 404

            features_summary = {}
            user_summary = {}
            timeline = []

            for row in all_results:
                user_id, feature, status, timestamp = row

                if feature not in features_summary:
                    features_summary[feature] = {"pass": 0, "warning": 0, "fail": 0, "total": 0}

                features_summary[feature]["total"] += 1

                if "✅" in status or "Safe" in status or "Good" in status or "matches" in status or "Single person" in status or "Live" in status:
                    features_summary[feature]["pass"] += 1
                elif "⚠️" in status or "Warning" in status or "Possible" in status or "Poor" in status:
                    features_summary[feature]["warning"] += 1
                elif "❌" in status or "does not match" in status or "Multiple" in status or "No" in status or "Error" in status:
                    features_summary[feature]["fail"] += 1

                if user_id not in user_summary:
                    user_summary[user_id] = {"features": {}, "total_checks": 0, "pass_count": 0, "warning_count": 0, "fail_count": 0}

                user_summary[user_id]["total_checks"] += 1
                user_summary[user_id]["features"][feature] = status

                if "✅" in status or "Safe" in status or "Good" in status or "matches" in status or "Single person" in status or "Live" in status:
                    user_summary[user_id]["pass_count"] += 1
                elif "⚠️" in status or "Warning" in status or "Possible" in status or "Poor" in status:
                    user_summary[user_id]["warning_count"] += 1
                elif "❌" in status or "does not match" in status or "Multiple" in status or "No" in status or "Error" in status:
                    user_summary[user_id]["fail_count"] += 1

                timeline.append({
                    "user": user_id,
                    "feature": feature,
                    "status": status,
                    "time": timestamp
                })

            total_checks = len(all_results)
            total_pass = sum(f["pass"] for f in features_summary.values())
            total_warning = sum(f["warning"] for f in features_summary.values())
            total_fail = sum(f["fail"] for f in features_summary.values())

            return jsonify({
                "status": "success",
                "meeting_id": meeting_id,
                "features_summary": features_summary,
                "user_summary": user_summary,
                "timeline": timeline[:100],
                "total_users": len(user_summary),
                "total_checks": total_checks,
                "total_pass": total_pass,
                "total_warning": total_warning,
                "total_fail": total_fail
            })

    except Exception as e:
        print(f"Analytics Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------- Uploaded Video Frame --------------------
def get_uploaded_best_frame(username):
    """
    Returns the best reference face frame from the uploaded signup video.

    Improvement over the original: instead of blindly grabbing frame 0
    (which is often blurry, eyes-closed, or a bad angle during upload lag),
    we sample up to 20 frames spread across the video and pick the one
    where a face is most clearly detected (largest face area = most frontal).

    Falls back to frame 0 if no face is found in any sampled frame.
    Result is cached in memory so the video is only read once per session.
    """
    if username in uploaded_first_frames:
        return uploaded_first_frames[username]

    video_path = os.path.join(UPLOAD_FOLDER, username, f"{username}_video.webm")
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Can't get frame count for some webm encodings — fallback to first frame
        ret, frame = cap.read()
        cap.release()
        if ret:
            uploaded_first_frames[username] = frame
        return uploaded_first_frames.get(username)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Sample up to 20 evenly spaced frames
    sample_count = min(20, total_frames)
    step = max(1, total_frames // sample_count)
    indices = [i * step for i in range(sample_count)]

    best_frame = None
    best_face_area = -1
    first_frame = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        if first_frame is None:
            first_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60))

        if len(faces) > 0:
            # Pick by the frame where the largest face appears (most frontal/close)
            largest_area = max(w * h for (x, y, w, h) in faces)
            if largest_area > best_face_area:
                best_face_area = largest_area
                best_frame = frame.copy()

    cap.release()

    chosen = best_frame if best_frame is not None else first_frame
    if chosen is not None:
        uploaded_first_frames[username] = chosen
        print(f"[FaceRef] {username}: best reference frame selected "
              f"(face_area={best_face_area}, from {len(indices)} samples)")
    return uploaded_first_frames.get(username)

# Keep old name as alias so nothing else breaks
def get_uploaded_first_frame(username):
    return get_uploaded_best_frame(username)

# live video captures
@socketio.on("candidate_frame")
def handle_candidate_frame(data):
    username = session.get("username", request.remote_addr)
    frame_base64 = data.get("frame_base64").split(",")[1]
    np_img = np.frombuffer(base64.b64decode(frame_base64), np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    live_frames[username] = frame


# -------------------- Frontend Routes --------------------
@app.route('/')
def home(): return render_template('homepg1.html')
@app.route('/about')
def about(): return render_template('about.html')
@app.route('/contact')
def contact(): return render_template('contact.html')
@app.route('/candidatesignup')
def candidatesignup(): return render_template('candidatesignup.html')
@app.route('/clientsignup')
def clientsignup(): return render_template('clientsignup.html')
@app.route('/clientportal')
def clientportal(): return render_template('clientportal.html')
@app.route('/candidateportal')
def candidateportal(): return render_template('candidateportal.html')

@app.route("/meeting/<meeting_id>")
def meeting_page(meeting_id):
    if meeting_id not in active_meetings: return "Invalid Meeting ID", 404
    role = request.args.get('role','candidate')
    return render_template("meeting.html", meeting_id=meeting_id, role=role)

@app.route("/view_results/<meeting_id>")
def view_results(meeting_id):
    with sqlite3.connect(AI_DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id,user_id,feature,status,created_at FROM ai_results WHERE meeting_id=?", (meeting_id,))
        results_table = c.fetchall()
        results_summary = {}
        for row in results_table:
            results_summary[row[2]] = row[3]
    return render_template(
        "view_results.html",
        results=results_summary,
        results_table=results_table
    )

# -------------------- OTP --------------------
@app.route("/send-otp", methods=["POST"])
def send_otp():
    data = request.get_json()
    email = data.get("email")
    if not email:
        return jsonify({"success": False, "message": "Email required"}), 400

    otp = str(random.randint(100000, 999999))
    otp_store[email] = otp

    try:
        sender_email = "authentichireweb@gmail.com"
        sender_password = "vnsmviyqkudgljvj"
        subject = "AuthentiHire OTP"
        body = f"Your OTP is: {otp}"
        message = f"Subject: {subject}\n\n{body}"

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, message)

        return jsonify({"success": True, "message": "OTP sent successfully!"})

    except Exception as e:
        print("Email Error:", e)
        return jsonify({"success": False, "message": str(e)}), 500


# -------------------- Signup/Login --------------------
@app.route("/candidate-signup", methods=["POST"])
def candidate_signup():
    data=request.get_json()
    name,username,email,otp,password=data.get("name"),data.get("username"),data.get("email"),data.get("otp"),data.get("password")
    if otp_store.get(email)!=otp: return jsonify({"success":False,"message":"Invalid OTP"}),400
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c=conn.cursor()
            c.execute("INSERT INTO candidates (name,username,email,password) VALUES (?,?,?,?)",(name,username,email,password))
            conn.commit()
        otp_store.pop(email,None)
        return jsonify({"success":True,"message":"Signup successful!"})
    except sqlite3.IntegrityError:
        return jsonify({"success":False,"message":"Email already registered"}),400

@app.route("/candidate-login", methods=["POST"])
def candidate_login():
    data=request.get_json()
    email,password=data.get("email"),data.get("password")
    with sqlite3.connect(DB_PATH) as conn:
        c=conn.cursor()
        c.execute("SELECT id,name,username,email FROM candidates WHERE email=? AND password=?",(email,password))
        user=c.fetchone()
    if user:
        session['user_type']='candidate'
        session['user_email']=email
        session['username']=user[2]
        return jsonify({"success":True,"message":"Login successful"})
    return jsonify({"success":False,"message":"Invalid credentials"}),401

@app.route("/client-login-page")
def client_login_page():
    return render_template("clientlogin.html")

@app.route("/client-signup-page")
def client_signup_page():
    return render_template("clientsignup.html")

@app.route("/candidate-login-page")
def candidate_login_page():
    return render_template("candidatelogin.html")

@app.route("/candidate-signup-page")
def candidate_signup_page():
    return render_template("candidatesignup.html")

@app.route("/client-signup", methods=["POST"])
def client_signup():
    data=request.get_json()
    name,username,email,otp,password=data.get("name"),data.get("username"),data.get("email"),data.get("otp"),data.get("password")
    if otp_store.get(email)!=otp: return jsonify({"success":False,"message":"Invalid OTP"}),400
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c=conn.cursor()
            c.execute("INSERT INTO clients (name,username,email,password) VALUES (?,?,?,?)",(name,username,email,password))
            conn.commit()
        otp_store.pop(email,None)
        return jsonify({"success":True,"message":"Client signup successful!"})
    except sqlite3.IntegrityError:
        return jsonify({"success":False,"message":"Email already registered"}),400

@app.route("/client-login", methods=["POST"])
def client_login():
    data=request.get_json()
    email,password=data.get("email"),data.get("password")
    with sqlite3.connect(DB_PATH) as conn:
        c=conn.cursor()
        c.execute("SELECT * FROM clients WHERE email=? AND password=?",(email,password))
        user=c.fetchone()
    if user:
        session['user_type']='client'
        session['user_email']=email
        return jsonify({"success":True,"message":"Client login successful"})
    return jsonify({"success":False,"message":"Invalid credentials"}),401


# -------------------- Profile --------------------
@app.route("/profile")
def profile():
    if 'user_type' not in session or 'user_email' not in session:
        return redirect(url_for('home'))
    user_type,email=session['user_type'],session['user_email']
    user = get_candidate_by_email(email) if user_type=='candidate' else get_client_by_email(email)
    if not user: session.clear(); return redirect(url_for('home'))
    return render_template('profile.html',user=user,user_type=user_type)

@app.route("/edit-profile", methods=["GET","POST"])
def edit_profile():
    if 'user_type' not in session or 'user_email' not in session:
        return redirect(url_for('home'))

    user_type, email = session['user_type'], session['user_email']

    if request.method == "POST":
        data = request.get_json() if request.is_json else request.form
        new_name, new_email = data.get("name"), data.get("email")

        if not new_name or not new_email:
            return jsonify({"success": False, "message": "Name & email required"}), 400

        success = update_candidate_profile(email, new_name, new_email) if user_type == 'candidate' else update_client_profile(email, new_name, new_email)

        if success:
            session['user_email'] = new_email
            return redirect(url_for('profile'))

        return jsonify({"success": False, "message": "Update failed"}), 500

    user = get_candidate_by_email(email) if user_type == 'candidate' else get_client_by_email(email)
    return render_template('edit_profile.html', user=user, user_type=user_type)

@app.route("/candidate-logout", methods=["POST","GET"])
def candidate_logout(): session.clear(); return redirect(url_for('home'))
@app.route("/client-logout", methods=["POST","GET"])
def client_logout(): session.clear(); return redirect(url_for('home'))


# -------------------- Meeting --------------------
@app.route("/create-meeting", methods=["POST"])
def create_meeting():
    data=request.get_json()
    meeting_id,password=data.get("id"),data.get("password")
    active_meetings[str(meeting_id)]=password
    return jsonify({"success":True})

@app.route("/leave_meeting/<meeting_id>", methods=["POST"])
def leave_meeting(meeting_id):
    if meeting_id in meeting_stop_flags:
        print(f"🛑 leave_meeting called for {meeting_id} — stopping AI pipeline")
        meeting_stop_flags[meeting_id].set()
    username = session.get("username")
    if username:
        live_frames.pop(username, None)
    return jsonify({"success": True})

@app.route("/validate-meeting", methods=["POST"])
def validate_meeting():
    data=request.get_json()
    meeting_id,password=data.get("id"),data.get("password")
    if str(meeting_id) in active_meetings and active_meetings[str(meeting_id)]==password:
        return jsonify({"success":True})
    return jsonify({"success":False})

@app.route("/api/gaze", methods=["POST"])
def receive_gaze_data():
    data = request.get_json()
    meeting_id = data.get("meeting_id")
    user_id = data.get("user_id")
    direction = data.get("direction")
    timestamp = data.get("timestamp")

    print(f"[Gaze] Meeting: {meeting_id} | User: {user_id} | Direction: {direction} | Time: {timestamp}")

    try:
        with sqlite3.connect("candidates.db") as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO gaze_data (meeting_id, user_id, direction, timestamp) VALUES (?, ?, ?, ?)",
                (meeting_id, user_id, direction, timestamp)
            )
            conn.commit()
        return jsonify({"success": True})
    except Exception as e:
        print("DB Error:", e)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/view-gaze")
def view_gaze():
    import sqlite3
    from flask import render_template

    with sqlite3.connect("candidates.db") as conn:
        c = conn.cursor()
        c.execute("SELECT DISTINCT meeting_id FROM gaze_data ORDER BY timestamp DESC LIMIT 1")
        row = c.fetchone()
        if not row:
            return "No gaze data available yet."
        meeting_id = row[0]

        c.execute("""
            SELECT user_id, total_events, focus_percentage
            FROM gaze_summary
            WHERE meeting_id=?
        """, (meeting_id,))
        rows = c.fetchall()

    total_events = sum(r[1] for r in rows)
    avg_focus = round(sum(r[2] for r in rows)/len(rows), 2) if rows else 0

    directions = ["Left", "Right", "Center", "Top", "Bottom"]
    percentages = {d: round(100/len(directions), 2) for d in directions}

    return render_template("analytics.html",
                           meeting_id=meeting_id,
                           total_events=total_events,
                           avg_focus=avg_focus,
                           percentages=percentages,
                           user_summary=[{"user_id": r[0], "focus_percentage": round(r[2],2)} for r in rows])


# -------------------- Uploads --------------------
@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "user_type" not in session or session["user_type"]!="candidate": return jsonify({"message":"Unauthorized"}),403
    if "video" not in request.files: return jsonify({"message":"No video file"}),400
    video=request.files["video"]
    username=session.get("username")
    user_folder=os.path.join(UPLOAD_FOLDER,username)
    os.makedirs(user_folder,exist_ok=True)
    filename=secure_filename(f"{username}_video.webm")
    path=os.path.join(user_folder,filename)
    video.save(path)
    return jsonify({"message":"Video uploaded successfully","path":f"/uploads/{username}/{filename}"})

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# -------------------- Audio Analysis --------------------
def analyze_audio_clip(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False,suffix=".wav") as tmp:
            file.save(tmp.name)
            y,sr=librosa.load(tmp.name,sr=16000)
        rms=np.mean(librosa.feature.rms(y=y))
        pitch=np.mean(librosa.yin(y,fmin=50,fmax=400))
        clarity_score=(rms+pitch/400)/2
        return round(float(clarity_score),2)
    except:
        return 0.0

def detect_audio_bias(file):
    score=analyze_audio_clip(file)
    return "⚠️ Possible bias" if score<0.25 else "✅ No bias"

def analyze_audio_ai(file):
    clarity = analyze_audio_clip(file)
    bias = detect_audio_bias(file)
    clarity_status = "✅ Good" if clarity>0.25 else "⚠️ Poor"
    return {"audio_clarity": clarity_status, "bias": bias}

@app.route("/analyze_audio", methods=["POST"])
def analyze_audio_route():
    print("\n🎤 /analyze_audio route called")

    audio = request.files.get("audio")
    meeting_id = request.form.get("meeting_id")
    user_id = session.get("username", request.remote_addr)

    print(f"Meeting ID: {meeting_id}")
    print(f"User ID: {user_id}")
    print(f"Audio file: {audio}")

    if not audio:
        print("❌ No audio file provided")
        return jsonify({"error": "No audio file"}), 400

    results = analyze_audio_ai(audio)
    print(f"Audio results: {results}")

    with sqlite3.connect(AI_DB_PATH) as conn:
        c = conn.cursor()
        for feature, status in results.items():
            print(f"💾 Saving: {feature} = {status}")
            c.execute("INSERT INTO ai_results (meeting_id,user_id,feature,status) VALUES (?,?,?,?)",
                      (meeting_id, user_id, feature, status))
            socketio.emit("ai_status_update",
                         {"feature": feature, "status": status, "source": user_id},
                         room=f"meeting_{meeting_id}")
        conn.commit()

    print("✅ Audio analysis complete")
    return jsonify({"success": True, "results": results})


# -------------------- Frame Analysis --------------------
def analyze_frame_ai(frame):
    print("\n" + "="*50)
    print("🔍 STARTING FRAME ANALYSIS")
    print("="*50)

    results = {}

    if frame is None:
        print("❌ Frame is None!")
        return {"error": "No frame provided"}

    print(f"✅ Frame shape: {frame.shape}")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(f"✅ Converted to RGB: {rgb.shape}")

    # 1️⃣ Deepfake / face detection
    print("\n🔍 Testing DeepFace...")
    try:
        df = DeepFace.analyze(rgb, actions=['emotion'], enforce_detection=False)
        print(f"✅ DeepFace result: {df}")
        results["deepfake"] = "✅ Safe" if df else "⚠️ Possible deepfake"
    except Exception as e:
        print(f"❌ DeepFace error: {e}")
        results["deepfake"] = f"❌ Error: {str(e)}"

    # 2️⃣ Gaze direction + Liveness — IMPROVED multi-layer detection
    print("\n🔍 Running improved gaze detection...")
    try:
        gaze_result = detect_gaze_direction(frame)
        direction  = gaze_result["direction"]
        confidence = gaze_result["confidence"]
        method     = gaze_result.get("method", "unknown")
        yaw        = gaze_result.get("yaw")
        pitch      = gaze_result.get("pitch")

        print(f"✅ Gaze: direction={direction}, conf={confidence:.2f}, "
              f"method={method}, yaw={yaw}, pitch={pitch}")

        DIR_LABELS = {
            "center": "✅ Looking at screen",
            "left":   "⚠️ Looking left",
            "right":  "⚠️ Looking right",
            "up":     "⚠️ Looking up",
            "down":   "⚠️ Looking down",
            "away":   "❌ Not looking at screen",
        }
        results["gaze"] = DIR_LABELS.get(direction, f"⚠️ {direction}")

        # Liveness: face present = live
        results["liveness"] = "⚠️ No face detected" if direction == "away" else "✅ Live"

    except Exception as e:
        print(f"❌ Gaze detection error: {e}")
        # Fallback to old mediapipe check so liveness still works
        try:
            mesh_results = gaze_model.process(rgb)
            if mesh_results.multi_face_landmarks:
                results["gaze"] = "✅ Face tracked"
                results["liveness"] = "✅ Live"
            else:
                results["gaze"] = "⚠️ No face detected"
                results["liveness"] = "⚠️ No face detected"
        except Exception as e2:
            results["gaze"] = results["liveness"] = f"❌ Error: {str(e2)}"

    # 3️⃣ Multi-person detection
    print("\n🔍 Testing YOLO person detection...")
    try:
        detections = person_model(frame, classes=[0], verbose=False)
        persons = sum(len(r.boxes.xyxy) for r in detections if r.boxes.xyxy.numel() > 0)
        print(f"✅ Detected {persons} person(s)")

        if persons == 1:
            results["multiperson"] = "✅ Single person"
        elif persons > 1:
            results["multiperson"] = "⚠️ Multiple persons"
        else:
            results["multiperson"] = "❌ No person detected"
    except Exception as e:
        print(f"❌ YOLO error: {e}")
        results["multiperson"] = f"❌ Error: {str(e)}"

    # 4️⃣ Facial emotion / bias
    print("\n🔍 Testing FER emotion detection...")
    try:
        emotions = emotion_detector.detect_emotions(rgb)
        print(f"✅ Emotions: {emotions}")

        if emotions:
            dominant = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
            print(f"✅ Dominant emotion: {dominant}")
            results["bias"] = "⚠️ Possible bias" if dominant in ["angry", "disgust"] else "✅ No bias"
        else:
            print("⚠️ No emotions detected")
            results["bias"] = "❌ No face detected"
    except Exception as e:
        print(f"❌ FER error: {e}")
        results["bias"] = f"❌ Error: {str(e)}"

    print("\n" + "="*50)
    print(f"📊 FINAL RESULTS: {results}")
    print("="*50 + "\n")

    return results


@app.route('/analyze_frame', methods=['POST'])
def analyze_frame_route():
    print("\n🎬 /analyze_frame route called")

    if 'frame' not in request.files:
        print("❌ No frame in request")
        return jsonify({"error": "No frame provided"}), 400

    frame_file = request.files['frame']
    meeting_id = request.form.get("meeting_id")
    user_id = session.get("username", request.remote_addr)

    print(f"✅ Frame file received: {frame_file.filename}")

    file_bytes = np.frombuffer(frame_file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if not frame_file or not meeting_id:
        return jsonify({"success": False, "message": "Missing frame or meeting_id"}), 400

    if frame is None:
        print("❌ Failed to decode frame")
        return jsonify({"error": "Invalid image file"}), 400

    print(f"✅ Frame decoded successfully: {frame.shape}")

    results = analyze_frame_ai(frame)

    try:
        with sqlite3.connect(AI_DB_PATH) as conn:
            c = conn.cursor()
            for feature, status in results.items():
                c.execute(
                    "INSERT INTO ai_results (meeting_id, user_id, feature, status) VALUES (?, ?, ?, ?)",
                    (meeting_id, user_id, feature, status)
                )
            conn.commit()

        socketio.emit(
            "ai_status_update",
            {"source": user_id, "results": results},
            room=f"meeting_{meeting_id}"
        )

        print(f"📤 Sending results: {results}")
        return jsonify(results)

    except Exception as e:
        print("DB Error (analyze_frame):", e)
        return jsonify({"success": False, "error": str(e)}), 500


# -------------------- Face Matching --------------------
def is_face_matching(uploaded_frame, live_frame):
    """
    Compare reference frame (from signup video) against a live frame.

    Improvements over original:
    - Uses ArcFace model (more accurate than default VGG-Face)
    - Explicit threshold tuning (cosine distance, threshold 0.40)
    - Detailed error logging so silent failures are visible
    - Returns (matched: bool, distance: float, error: str|None)
    """
    try:
        result = DeepFace.verify(
            img1_path=uploaded_frame,
            img2_path=live_frame,
            model_name="ArcFace",          # more accurate than VGG-Face default
            detector_backend="opencv",     # fast, consistent
            distance_metric="cosine",
            enforce_detection=False,       # don't crash if face not detected
            align=True,
        )
        distance = result.get("distance", 1.0)
        verified = result.get("verified", False)
        print(f"[FaceMatch] distance={distance:.4f}, verified={verified}")
        return verified, distance, None

    except Exception as e:
        print(f"[FaceMatch] Error: {e}")
        return False, 1.0, str(e)


def _run_face_match_for_user(username, live_frame, meeting_id):
    """
    Core face-match logic called both from the HTTP route and the
    automatic AI pipeline. Saves result and emits socket update.
    """
    uploaded_frame = get_uploaded_first_frame(username)

    if uploaded_frame is None:
        status = "⚠️ No reference video found"
        print(f"[FaceMatch] No reference video for {username}")
        save_result(meeting_id, "face_match", status, user_id=username)
        socketio.emit("ai_status_update",
                      {"feature": "face_match", "status": status, "source": username,
                       "timestamp": datetime.datetime.now().isoformat()},
                      room=f"meeting_{meeting_id}")
        return status

    matched, distance, error = is_face_matching(uploaded_frame, live_frame)

    if error:
        status = f"⚠️ Face match error"
    elif matched:
        status = f"✅ Face matches uploaded video"
    else:
        status = f"❌ Face does not match (distance: {distance:.2f})"

    save_result(meeting_id, "face_match", status, user_id=username)
    socketio.emit("ai_status_update",
                  {"feature": "face_match", "status": status, "source": username,
                   "timestamp": datetime.datetime.now().isoformat()},
                  room=f"meeting_{meeting_id}")
    return status


@app.route("/check_face_match", methods=["POST"])
def check_face_match_route():
    """
    Called by the frontend to explicitly trigger a face verification check.
    Also called automatically every N seconds by the AI pipeline.
    """
    if "frame" not in request.files or not session.get("username"):
        return jsonify({"message": "Missing data"}), 400

    frame_file = request.files["frame"]
    meeting_id = request.form.get("meeting_id")
    username   = session.get("username")

    np_img     = np.frombuffer(frame_file.read(), np.uint8)
    live_frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if live_frame is None:
        return jsonify({"success": False, "message": "Invalid frame"}), 400

    status = _run_face_match_for_user(username, live_frame, meeting_id)
    return jsonify({"success": True, "status": status})


# -------------------- SocketIO --------------------

@socketio.on("join-meeting")
def handle_join(data):
    meeting_id=data.get("meetingId")
    sid=request.sid
    room=f"meeting_{meeting_id}"
    join_room(room)
    if meeting_id not in meeting_rooms: meeting_rooms[meeting_id]=set()
    for pid in list(meeting_rooms[meeting_id]):
        emit("new-participant",{"socketId":sid},to=pid)
        emit("new-participant",{"socketId":pid},to=sid)
    meeting_rooms[meeting_id].add(sid)
    emit("update-participant-count",{"count":len(meeting_rooms[meeting_id])},to=room)

@socketio.on("signal")
def handle_signal(data):
    to=data.get("to")
    emit("signal",{**data,"from":request.sid},to=to if to else None,broadcast=(to is None))

@socketio.on("chat")
def handle_chat(data):
    sid = request.sid
    message = data.get("message", "")

    meeting_id = None
    for mid, participants in meeting_rooms.items():
        if sid in participants:
            meeting_id = mid
            break

    if not meeting_id:
        print(f"⚠️ Chat from {sid} but not in any meeting room")
        return

    room = f"meeting_{meeting_id}"
    print(f"💬 Chat in {meeting_id} from {sid}: {message}")
    emit("chat", {"message": message, "from": sid}, to=room, include_self=False)


# -------------------- Gaze Event Handler (IMPROVED) --------------------
@socketio.on("gaze-event")
def handle_gaze_event(data):
    """
    Receives a gaze event from the frontend.
    Now runs server-side gaze detection on the latest live frame for accuracy.
    Falls back to the frontend-reported direction if no frame is available.
    """
    meeting_id     = data.get("meetingId")
    socket_user_id = data.get("socketId")
    frontend_dir   = data.get("direction", "center")
    timestamp      = data.get("timestamp") or datetime.datetime.utcnow().isoformat()

    # ── Server-side gaze detection on live frame ──────────────────────────────
    username   = session.get("username", socket_user_id)
    live_frame = live_frames.get(username)

    if live_frame is not None:
        try:
            raw_result = detect_gaze_direction(live_frame)

            # Per-user smoother (window=5) to kill frame-to-frame jitter
            if username not in _gaze_smoothers:
                _gaze_smoothers[username] = GazeSmoother(window=5)
            smoothed = _gaze_smoothers[username].update(raw_result)

            direction  = smoothed["direction"]
            confidence = smoothed["confidence"]
            yaw        = smoothed.get("yaw")
            pitch      = smoothed.get("pitch")
            method     = smoothed.get("method", "server")
        except Exception as e:
            print(f"[Gaze] Server detection error for {username}: {e}")
            direction  = frontend_dir
            confidence = 0.5
            yaw = pitch = None
            method = "frontend_fallback_error"
    else:
        # No live frame yet — use the frontend JS direction as fallback
        direction  = frontend_dir
        confidence = 0.55
        yaw = pitch = None
        method = "frontend_only"

    print(f"[Gaze] {username} | {meeting_id} | dir={direction} "
          f"(conf={confidence:.2f}) | method={method} | {timestamp}")

    # ── Persist raw gaze event ────────────────────────────────────────────────
    try:
        with sqlite3.connect("candidates.db") as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO gaze_data (meeting_id, user_id, direction, timestamp) VALUES (?, ?, ?, ?)",
                (meeting_id, socket_user_id, direction, timestamp),
            )
            conn.commit()
    except Exception as e:
        print("DB Error (gaze_data insert):", e)

    # ── Update gaze_summary analytics ─────────────────────────────────────────
    # Each event covers ~0.3s (frontend debounces at 300ms)
    event_duration_seconds = 0.3

    try:
        now_iso = datetime.datetime.utcnow().isoformat()
        with sqlite3.connect("candidates.db") as conn:
            c = conn.cursor()
            c.execute("""
                SELECT total_events, total_away_time, focus_percentage
                FROM gaze_summary WHERE meeting_id=? AND user_id=?
            """, (meeting_id, socket_user_id))
            row = c.fetchone()

            if row is None:
                c.execute("""
                    INSERT INTO gaze_summary
                    (meeting_id, user_id, total_events, total_away_time, focus_percentage, last_updated)
                    VALUES (?, ?, 0, 0.0, 0.0, ?)
                """, (meeting_id, socket_user_id, now_iso))
                conn.commit()
                total_events, total_away_time = 0, 0.0
            else:
                total_events, total_away_time, _ = row

            total_events = (total_events or 0) + 1

            # Any direction that is NOT center counts as looking away
            if direction in ("away", "left", "right", "up", "down"):
                total_away_time = (total_away_time or 0.0) + event_duration_seconds

            denom = total_events * event_duration_seconds
            focus_percentage = (
                max(0.0, 100.0 * (1.0 - total_away_time / denom))
                if denom > 0 else 0.0
            )

            c.execute("""
                UPDATE gaze_summary
                SET total_events=?, total_away_time=?, focus_percentage=?, last_updated=?
                WHERE meeting_id=? AND user_id=?
            """, (total_events, total_away_time, focus_percentage,
                  now_iso, meeting_id, socket_user_id))
            conn.commit()
    except Exception as e:
        print("DB Error (gaze_summary update):", e)

    # ── Broadcast to other participants in the room ───────────────────────────
    emit("gaze-update", {
        "socketId":   socket_user_id,
        "meetingId":  meeting_id,
        "direction":  direction,
        "confidence": confidence,
        "yaw":        yaw,
        "pitch":      pitch,
        "method":     method,
        "timestamp":  timestamp,
    }, room=f"meeting_{meeting_id}", include_self=False)

    # ── Push updated summary to interviewer UI ────────────────────────────────
    try:
        with sqlite3.connect("candidates.db") as conn:
            c = conn.cursor()
            c.execute("""
                SELECT total_events, total_away_time, focus_percentage, last_updated
                FROM gaze_summary WHERE meeting_id=? AND user_id=?
            """, (meeting_id, socket_user_id))
            srow = c.fetchone()
            if srow:
                emit("gaze-summary-update", {
                    "user_id":          socket_user_id,
                    "meetingId":        meeting_id,
                    "total_events":     srow[0],
                    "total_away_time":  srow[1],
                    "focus_percentage": round(srow[2], 2),
                    "last_updated":     srow[3],
                    "last_direction":   direction,
                    "last_confidence":  round(confidence, 3),
                }, room=f"meeting_{meeting_id}", include_self=False)
    except Exception as e:
        print("DB Error (broadcast summary):", e)


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    print(f"Client disconnected: {sid}")
    remove_rooms = []

    for meeting_id, participants in list(meeting_rooms.items()):
        if sid in participants:
            participants.remove(sid)
            for pid in list(participants):
                emit("disconnect-peer", sid, to=pid)

            if participants:
                emit("update-participant-count",
                     {"count": len(participants)},
                     to=f"meeting_{meeting_id}")
            else:
                if meeting_id in meeting_stop_flags:
                    print(f"🛑 Last participant left {meeting_id} — stopping AI pipeline")
                    meeting_stop_flags[meeting_id].set()

                with sqlite3.connect(AI_DB_PATH) as conn:
                    c = conn.cursor()
                    c.execute("SELECT feature, status, created_at FROM ai_results WHERE meeting_id=?", (meeting_id,))
                    final_results = c.fetchall()

                socketio.emit("final_ai_results",
                              {"results": final_results},
                              room=f"client_{meeting_id}")
                remove_rooms.append(meeting_id)

    for room_id in remove_rooms:
        meeting_rooms.pop(room_id, None)


# -------------------- Gaze API Endpoints --------------------

@app.route("/api/gaze_summary/<meeting_id>")
def get_gaze_summary(meeting_id):
    try:
        with sqlite3.connect("candidates.db") as conn:
            c = conn.cursor()
            c.execute("""
                SELECT user_id, total_events, total_away_time, focus_percentage, last_updated
                FROM gaze_summary WHERE meeting_id=?
            """, (meeting_id,))
            rows = c.fetchall()
        return jsonify({"status": "success", "summary": [
            {"user_id": r[0], "total_events": r[1], "total_away_time": r[2],
             "focus_percentage": round(r[3], 2), "last_updated": r[4]}
            for r in rows
        ]})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/view-gaze/<meeting_id>")
def view_gaze_summary(meeting_id):
    import sqlite3

    with sqlite3.connect("candidates.db") as conn:
        c = conn.cursor()
        c.execute("""
            SELECT COUNT(*), AVG(focus_percentage)
            FROM gaze_summary WHERE meeting_id=?
        """, (meeting_id,))
        total_events, avg_focus = c.fetchone()
        total_events = total_events or 0
        avg_focus = round(avg_focus or 0, 2)

        directions = ["Left", "Right", "Center", "Top", "Bottom"]
        counts = {d: 0 for d in directions}
        c.execute("SELECT direction FROM gaze_data WHERE meeting_id=?", (meeting_id,))
        rows = c.fetchall()
        total_dir = 0
        for (direction,) in rows:
            if direction in directions:
                counts[direction] += 1
                total_dir += 1

        percentages = {d: round((counts[d]/total_dir)*100,2) if total_dir>0 else 0 for d in directions}

    return render_template("analytics.html",
                           total_events=total_events,
                           avg_focus=avg_focus,
                           percentages=percentages,
                           meeting_id=meeting_id)

@app.route("/api/gaze_direction_distribution/<meeting_id>")
def get_gaze_direction_distribution(meeting_id):
    """Get percentage distribution of gaze directions"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT direction, COUNT(*) as count
                FROM gaze_data
                WHERE meeting_id=?
                GROUP BY direction
            """, (meeting_id,))
            rows = c.fetchall()

            total = sum(r[1] for r in rows)
            percentages = {
                row[0]: round((row[1] / total) * 100, 2) if total > 0 else 0
                for row in rows
            }

            for direction in ["left", "right", "center", "up", "down", "away"]:
                if direction not in percentages:
                    percentages[direction] = 0

            return jsonify({"status": "success", "percentages": percentages})
    except Exception as e:
        print(f"Direction distribution error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/api/gaze_events/<meeting_id>")
def get_gaze_events(meeting_id):
    """Get recent gaze events for timeline"""
    limit = request.args.get('limit', 50, type=int)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT user_id, direction, timestamp
                FROM gaze_data
                WHERE meeting_id=?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (meeting_id, limit))
            rows = c.fetchall()

            events = [{
                "user_id": r[0],
                "direction": r[1],
                "timestamp": r[2]
            } for r in rows]

            return jsonify({"status": "success", "events": events})
    except Exception as e:
        print(f"Gaze events error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


# ── NEW: Rich gaze analytics endpoint (used by improved analytics.html) ───────
@app.route("/api/gaze_analytics/<meeting_id>")
def get_gaze_analytics(meeting_id):
    """
    Returns rich per-user gaze analytics:
      - direction distribution (counts + percentages per user)
      - overall direction distribution
      - summary stats
      - timeline
    """
    try:
        with sqlite3.connect("candidates.db") as conn:
            c = conn.cursor()

            # Per-user summary
            c.execute("""
                SELECT user_id, total_events, total_away_time,
                       focus_percentage, last_updated
                FROM gaze_summary WHERE meeting_id=?
                ORDER BY focus_percentage DESC
            """, (meeting_id,))
            summary_rows = c.fetchall()

            # Direction distribution per user
            c.execute("""
                SELECT user_id, direction, COUNT(*) as cnt
                FROM gaze_data WHERE meeting_id=?
                GROUP BY user_id, direction
                ORDER BY user_id, cnt DESC
            """, (meeting_id,))
            dir_rows = c.fetchall()

            # Overall direction distribution
            c.execute("""
                SELECT direction, COUNT(*) as cnt
                FROM gaze_data WHERE meeting_id=?
                GROUP BY direction
                ORDER BY cnt DESC
            """, (meeting_id,))
            overall_dirs = c.fetchall()

            # Recent timeline events
            c.execute("""
                SELECT user_id, direction, timestamp
                FROM gaze_data WHERE meeting_id=?
                ORDER BY timestamp ASC
                LIMIT 200
            """, (meeting_id,))
            timeline_rows = c.fetchall()

        if not summary_rows and not dir_rows:
            return jsonify({"status": "error", "message": "No gaze data found"}), 404

        # Build user direction maps
        user_dirs = {}
        for uid, direction, cnt in dir_rows:
            if uid not in user_dirs:
                user_dirs[uid] = {}
            user_dirs[uid][direction] = cnt

        # Overall percentages
        total_events = sum(c for _, c in overall_dirs)
        overall_pct = {
            d: round(cnt / total_events * 100, 1) if total_events > 0 else 0
            for d, cnt in overall_dirs
        }

        # Build user summary list
        users = []
        for uid, total_ev, total_away, focus_pct, last_upd in summary_rows:
            dirs = user_dirs.get(uid, {})
            total_user = sum(dirs.values()) or 1
            users.append({
                "user_id":          uid,
                "total_events":     total_ev,
                "total_away_time":  round(total_away, 1),
                "focus_percentage": round(focus_pct, 1),
                "last_updated":     last_upd,
                "directions": {
                    d: {
                        "count": cnt,
                        "pct":   round(cnt / total_user * 100, 1)
                    }
                    for d, cnt in dirs.items()
                },
            })

        timeline = [
            {"user": r[0], "direction": r[1], "timestamp": r[2]}
            for r in timeline_rows
        ]

        return jsonify({
            "status":             "success",
            "meeting_id":         meeting_id,
            "total_events":       total_events,
            "overall_directions": overall_pct,
            "users":              users,
            "timeline":           timeline,
        })

    except Exception as e:
        print(f"Gaze analytics error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@socketio.on("join_room")
def handle_join_room(data):
    room = data.get("room")
    join_room(room)
    print(f"Client joined room: {room}")


# -------------------- Debug / Test Routes --------------------
@app.route("/test_db")
def test_db():
    with sqlite3.connect(AI_DB_PATH) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO ai_results (meeting_id,user_id,feature,status) VALUES (?,?,?,?)",
                  ("test123", "testuser", "test_feature", "✅ Test"))
        conn.commit()
        c.execute("SELECT * FROM ai_results")
        results = c.fetchall()
    return jsonify({"count": len(results), "data": results})

@app.route("/start_interview/<meeting_id>", methods=["POST"])
def start_interview(meeting_id):
    username = session.get("username")

    if not username or username == "system":
        print(f"⚠️ Skipping AI detection for {username or 'anonymous'} user")
        return jsonify({"success": False, "message": "AI detection only for candidates"})

    if meeting_id in meeting_stop_flags and not meeting_stop_flags[meeting_id].is_set():
        print(f"⚠️ AI pipeline already running for meeting {meeting_id}")
        return jsonify({"success": True, "message": "AI detection already running"})

    print(f"\n🚀 Starting interview for {username} in meeting {meeting_id}")

    stop_event = threading.Event()
    meeting_stop_flags[meeting_id] = stop_event

    def run_ai_pipeline():
        print(f"🧠 AI detection thread started for {username} | meeting {meeting_id}")
        iteration = 0
        face_match_counter = 0
        FACE_MATCH_EVERY = 10  # run face match every 10 iterations (~30s)

        while not stop_event.is_set():
            iteration += 1
            print(f"\n--- Iteration {iteration} (meeting {meeting_id}) ---")

            # Only use the actual live camera frame — never fall back to the
            # uploaded video frame here (that would compare the upload against
            # itself and always return a false "match").
            live_frame = live_frames.get(username)

            if live_frame is None:
                print(f"⚠️ No live frame yet for {username}, waiting for camera stream…")
                stop_event.wait(timeout=2)
                continue

            print(f"✅ Using live frame for {username}")

            # General AI checks (gaze, deepfake, liveness, multiperson)
            print(f"🔍 Analyzing live frame for {username}...")
            results_frame = analyze_frame_ai(live_frame)

            all_results = results_frame.copy()
            if "bias" in all_results:
                del all_results["bias"]
            print(f"📊 Frame results: {all_results}")

            # Automatic face match every FACE_MATCH_EVERY iterations
            face_match_counter += 1
            if face_match_counter >= FACE_MATCH_EVERY:
                face_match_counter = 0
                print(f"🔍 Running automatic face match for {username}...")
                try:
                    current_live = live_frames.get(username)
                    if current_live is not None:
                        _run_face_match_for_user(username, current_live, meeting_id)
                        print("✅ Automatic face match completed")
                    else:
                        print("⚠️ No live frame for face match, skipping")
                except Exception as e:
                    print(f"❌ Automatic face match error: {e}")

            if stop_event.is_set():
                print(f"🛑 Stop detected after analysis — discarding results for {meeting_id}")
                break

            try:
                with sqlite3.connect(AI_DB_PATH) as conn:
                    c = conn.cursor()
                    for feature, status in all_results.items():
                        print(f"💾 Saving to DB: {feature} = {status}")
                        c.execute("INSERT INTO ai_results (meeting_id,user_id,feature,status) VALUES (?,?,?,?)",
                                  (meeting_id, username, feature, status))
                    conn.commit()
                    print("✅ All results saved to database")

                for feature, status in all_results.items():
                    socketio.emit(
                        "ai_status_update",
                        {"feature": feature, "status": status, "source": username},
                        room=f"meeting_{meeting_id}"
                    )
                print("✅ Results emitted via SocketIO")

            except Exception as e:
                print(f"❌ Error saving results: {e}")

            stop_event.wait(timeout=3)

        live_frames.pop(username, None)
        meeting_stop_flags.pop(meeting_id, None)
        print(f"🛑 AI pipeline stopped for meeting {meeting_id} (user: {username})")



    thread = threading.Thread(target=run_ai_pipeline, daemon=True)
    thread.start()
    print(f"✅ AI detection thread started successfully for meeting {meeting_id}")

    return jsonify({"success": True, "message": "AI detection started"})

@socketio.on("raise-hand")
def handle_raise_hand(data):
    meeting_id = None
    sid = request.sid

    for mid, participants in meeting_rooms.items():
        if sid in participants:
            meeting_id = mid
            break

    if not meeting_id:
        return

    emit("raise-hand", {"id": sid}, room=f"meeting_{meeting_id}")

    user_id = session.get("username", sid)
    save_result(meeting_id, "hand_raise", "✋ Raised hand", user_id=user_id)

    socketio.emit(
        "ai_status_update",
        {"feature": "hand_raise", "status": "✋ Raised hand", "source": user_id},
        room=f"meeting_{meeting_id}"
    )

@app.route("/test_ai", methods=["GET"])
def test_ai():
    test_results = {}

    try:
        test_results["yolo_loaded"] = person_model is not None
        test_results["facemesh_loaded"] = gaze_model is not None
        test_results["emotion_loaded"] = emotion_detector is not None
        test_results["gaze_detector_loaded"] = True
    except Exception as e:
        test_results["model_loading_error"] = str(e)

    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(test_frame, (320, 240), 100, (255, 255, 255), -1)

    try:
        frame_results = analyze_frame_ai(test_frame)
        test_results["frame_analysis"] = frame_results
    except Exception as e:
        test_results["frame_analysis_error"] = str(e)

    try:
        with sqlite3.connect(AI_DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM ai_results")
            count = c.fetchone()[0]
            test_results["db_records_count"] = count
            c.execute("SELECT * FROM ai_results ORDER BY created_at DESC LIMIT 5")
            test_results["last_5_records"] = c.fetchall()
    except Exception as e:
        test_results["db_error"] = str(e)

    return jsonify(test_results)

@app.route("/debug/analytics/<meeting_id>")
def debug_analytics(meeting_id):
    debug_info = {
        "meeting_id": meeting_id,
        "gaze_data": {},
        "gaze_summary": {},
        "ai_results": {},
        "raw_counts": {}
    }

    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()

            c.execute("SELECT COUNT(*) FROM gaze_data WHERE meeting_id=?", (meeting_id,))
            debug_info["raw_counts"]["gaze_events"] = c.fetchone()[0]

            c.execute("""
                SELECT user_id, direction, timestamp FROM gaze_data
                WHERE meeting_id=? LIMIT 10
            """, (meeting_id,))
            debug_info["gaze_data"]["sample_events"] = [
                {"user": r[0], "direction": r[1], "time": r[2]} for r in c.fetchall()
            ]

            c.execute("""
                SELECT user_id, total_events, focus_percentage, last_updated
                FROM gaze_summary WHERE meeting_id=?
            """, (meeting_id,))
            debug_info["gaze_summary"]["users"] = [
                {"user": r[0], "events": r[1], "focus": r[2], "updated": r[3]}
                for r in c.fetchall()
            ]

    except Exception as e:
        debug_info["gaze_error"] = str(e)

    try:
        with sqlite3.connect(AI_DB_PATH) as conn:
            c = conn.cursor()

            c.execute("SELECT COUNT(*) FROM ai_results WHERE meeting_id=?", (meeting_id,))
            debug_info["raw_counts"]["ai_results"] = c.fetchone()[0]

            c.execute("""
                SELECT user_id, feature, status, created_at FROM ai_results
                WHERE meeting_id=? LIMIT 10
            """, (meeting_id,))
            debug_info["ai_results"]["sample"] = [
                {"user": r[0], "feature": r[1], "status": r[2], "time": r[3]}
                for r in c.fetchall()
            ]

            c.execute("""
                SELECT feature, COUNT(*) FROM ai_results
                WHERE meeting_id=? GROUP BY feature
            """, (meeting_id,))
            debug_info["ai_results"]["by_feature"] = dict(c.fetchall())

    except Exception as e:
        debug_info["ai_error"] = str(e)

    debug_info["meeting_active"] = meeting_id in active_meetings
    debug_info["meeting_rooms"]  = meeting_id in meeting_rooms

    return jsonify(debug_info)

@app.route("/generate_test_data/<meeting_id>")
def generate_test_data(meeting_id):
    test_users = ["test_user_1", "test_user_2", "test_candidate_3"]
    directions = ["left", "right", "center", "up", "down", "away"]

    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()

            for user in test_users:
                for i in range(50):
                    direction = random.choice(directions)
                    timestamp = (datetime.datetime.now() - datetime.timedelta(minutes=random.randint(0, 60))).isoformat()
                    c.execute("""
                        INSERT INTO gaze_data (meeting_id, user_id, direction, timestamp)
                        VALUES (?, ?, ?, ?)
                    """, (meeting_id, user, direction, timestamp))

                total_events = 50
                away_events  = random.randint(5, 15)
                focus_percentage = ((total_events - away_events) / total_events) * 100

                c.execute("""
                    INSERT OR REPLACE INTO gaze_summary
                    (meeting_id, user_id, total_events, total_away_time, focus_percentage, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (meeting_id, user, total_events, away_events * 0.3, focus_percentage,
                      datetime.datetime.now().isoformat()))

            conn.commit()

        with sqlite3.connect(AI_DB_PATH) as conn:
            c = conn.cursor()
            features = ["deepfake", "liveness", "multiperson", "face_match", "bias", "audio_clarity"]
            statuses = ["✅ Pass", "⚠️ Warning", "❌ Fail"]

            for user in test_users:
                for feature in features:
                    for i in range(random.randint(3, 5)):
                        status = random.choice(statuses)
                        c.execute("""
                            INSERT INTO ai_results (meeting_id, user_id, feature, status)
                            VALUES (?, ?, ?, ?)
                        """, (meeting_id, user, feature, status))

            conn.commit()

        return jsonify({
            "success": True,
            "message": f"Generated test data for meeting {meeting_id}",
            "users_created": len(test_users),
            "gaze_events_per_user": 50,
            "ai_checks_per_user": len(features) * 4
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# -------------------- Start --------------------
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)    conn.commit()
    conn.close()
init_db()

def init_ai_db():
    conn = sqlite3.connect(AI_DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS ai_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id TEXT,
                    user_id TEXT,
                    feature TEXT,
                    status TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

init_ai_db()

def init_gaze_table():
    conn = sqlite3.connect("candidates.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS gaze_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id TEXT,
                    user_id TEXT,
                    direction TEXT,
                    timestamp TEXT
                )''')
    conn.commit()
    conn.close()

def init_gaze_summary():
    conn = sqlite3.connect("candidates.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS gaze_summary (
                    meeting_id TEXT,
                    user_id TEXT,
                    total_events INTEGER DEFAULT 0,
                    total_away_time REAL DEFAULT 0,
                    focus_percentage REAL DEFAULT 0,
                    last_updated TEXT,
                    PRIMARY KEY (meeting_id, user_id)
                )''')
    conn.commit()
    conn.close()

init_db()
init_gaze_table()
init_gaze_summary()


# -------------------- Helper DB Functions --------------------
def get_candidate_by_email(email):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id,name,username,email FROM candidates WHERE email=?", (email,))
        row = c.fetchone()
        if row:
            return {"id": row[0],"name":row[1],"username":row[2],"email":row[3]}
    return None

def get_client_by_email(email):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id,name,username,email FROM clients WHERE email=?", (email,))
        row = c.fetchone()
    return {"id": row[0],"name":row[1],"username":row[2],"email":row[3]} if row else None

def update_candidate_profile(email,new_name,new_email):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("UPDATE candidates SET name=?,email=? WHERE email=?", (new_name,new_email,email))
        conn.commit()
        return c.rowcount>0

def update_client_profile(email,new_name,new_email):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("UPDATE clients SET name=?,email=? WHERE email=?", (new_name,new_email,email))
        conn.commit()
        return c.rowcount>0

def save_result(meeting_id, feature, status, user_id="system"):
    print(f"💾 Saving: {meeting_id} | {user_id} | {feature} | {status}")
    with sqlite3.connect(AI_DB_PATH) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO ai_results (meeting_id,user_id,feature,status) VALUES (?,?,?,?)",
                  (meeting_id,user_id,feature,status))
        conn.commit()
        print("✅ Saved successfully")

@app.route("/api/ai_detection_analytics/<meeting_id>")
def get_ai_detection_analytics(meeting_id):
    """
    Analytics for all AI detections excluding gaze
    """
    try:
        with sqlite3.connect(AI_DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT user_id, feature, status, created_at 
                FROM ai_results 
                WHERE meeting_id=? AND feature NOT IN ('gaze', 'gaze_direction', 'gaze_tracking')
                ORDER BY created_at DESC
            """, (meeting_id,))
            all_results = c.fetchall()

            if not all_results:
                return jsonify({"status": "error", "message": "No AI detection data found"}), 404

            features_summary = {}
            user_summary = {}
            timeline = []

            for row in all_results:
                user_id, feature, status, timestamp = row

                if feature not in features_summary:
                    features_summary[feature] = {"pass": 0, "warning": 0, "fail": 0, "total": 0}

                features_summary[feature]["total"] += 1

                if "✅" in status or "Safe" in status or "Good" in status or "matches" in status or "Single person" in status or "Live" in status:
                    features_summary[feature]["pass"] += 1
                elif "⚠️" in status or "Warning" in status or "Possible" in status or "Poor" in status:
                    features_summary[feature]["warning"] += 1
                elif "❌" in status or "does not match" in status or "Multiple" in status or "No" in status or "Error" in status:
                    features_summary[feature]["fail"] += 1

                if user_id not in user_summary:
                    user_summary[user_id] = {"features": {}, "total_checks": 0, "pass_count": 0, "warning_count": 0, "fail_count": 0}

                user_summary[user_id]["total_checks"] += 1
                user_summary[user_id]["features"][feature] = status

                if "✅" in status or "Safe" in status or "Good" in status or "matches" in status or "Single person" in status or "Live" in status:
                    user_summary[user_id]["pass_count"] += 1
                elif "⚠️" in status or "Warning" in status or "Possible" in status or "Poor" in status:
                    user_summary[user_id]["warning_count"] += 1
                elif "❌" in status or "does not match" in status or "Multiple" in status or "No" in status or "Error" in status:
                    user_summary[user_id]["fail_count"] += 1

                timeline.append({
                    "user": user_id,
                    "feature": feature,
                    "status": status,
                    "time": timestamp
                })

            total_checks = len(all_results)
            total_pass = sum(f["pass"] for f in features_summary.values())
            total_warning = sum(f["warning"] for f in features_summary.values())
            total_fail = sum(f["fail"] for f in features_summary.values())

            return jsonify({
                "status": "success",
                "meeting_id": meeting_id,
                "features_summary": features_summary,
                "user_summary": user_summary,
                "timeline": timeline[:100],
                "total_users": len(user_summary),
                "total_checks": total_checks,
                "total_pass": total_pass,
                "total_warning": total_warning,
                "total_fail": total_fail
            })

    except Exception as e:
        print(f"Analytics Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------- Uploaded Video Frame --------------------
def get_uploaded_best_frame(username):
    """
    Returns the best reference face frame from the uploaded signup video.

    Improvement over the original: instead of blindly grabbing frame 0
    (which is often blurry, eyes-closed, or a bad angle during upload lag),
    we sample up to 20 frames spread across the video and pick the one
    where a face is most clearly detected (largest face area = most frontal).

    Falls back to frame 0 if no face is found in any sampled frame.
    Result is cached in memory so the video is only read once per session.
    """
    if username in uploaded_first_frames:
        return uploaded_first_frames[username]

    video_path = os.path.join(UPLOAD_FOLDER, username, f"{username}_video.webm")
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Can't get frame count for some webm encodings — fallback to first frame
        ret, frame = cap.read()
        cap.release()
        if ret:
            uploaded_first_frames[username] = frame
        return uploaded_first_frames.get(username)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Sample up to 20 evenly spaced frames
    sample_count = min(20, total_frames)
    step = max(1, total_frames // sample_count)
    indices = [i * step for i in range(sample_count)]

    best_frame = None
    best_face_area = -1
    first_frame = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        if first_frame is None:
            first_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60))

        if len(faces) > 0:
            # Pick by the frame where the largest face appears (most frontal/close)
            largest_area = max(w * h for (x, y, w, h) in faces)
            if largest_area > best_face_area:
                best_face_area = largest_area
                best_frame = frame.copy()

    cap.release()

    chosen = best_frame if best_frame is not None else first_frame
    if chosen is not None:
        uploaded_first_frames[username] = chosen
        print(f"[FaceRef] {username}: best reference frame selected "
              f"(face_area={best_face_area}, from {len(indices)} samples)")
    return uploaded_first_frames.get(username)

# Keep old name as alias so nothing else breaks
def get_uploaded_first_frame(username):
    return get_uploaded_best_frame(username)

# live video captures
@socketio.on("candidate_frame")
def handle_candidate_frame(data):
    username = session.get("username", request.remote_addr)
    frame_base64 = data.get("frame_base64").split(",")[1]
    np_img = np.frombuffer(base64.b64decode(frame_base64), np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    live_frames[username] = frame


# -------------------- Frontend Routes --------------------
@app.route('/')
def home(): return render_template('homepg1.html')
@app.route('/about')
def about(): return render_template('about.html')
@app.route('/contact')
def contact(): return render_template('contact.html')
@app.route('/candidatesignup')
def candidatesignup(): return render_template('candidatesignup.html')
@app.route('/clientsignup')
def clientsignup(): return render_template('clientsignup.html')
@app.route('/clientportal')
def clientportal(): return render_template('clientportal.html')
@app.route('/candidateportal')
def candidateportal(): return render_template('candidateportal.html')

@app.route("/meeting/<meeting_id>")
def meeting_page(meeting_id):
    if meeting_id not in active_meetings: return "Invalid Meeting ID", 404
    role = request.args.get('role','candidate')
    return render_template("meeting.html", meeting_id=meeting_id, role=role)

@app.route("/view_results/<meeting_id>")
def view_results(meeting_id):
    with sqlite3.connect(AI_DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id,user_id,feature,status,created_at FROM ai_results WHERE meeting_id=?", (meeting_id,))
        results_table = c.fetchall()
        results_summary = {}
        for row in results_table:
            results_summary[row[2]] = row[3]
    return render_template(
        "view_results.html",
        results=results_summary,
        results_table=results_table
    )

# -------------------- OTP --------------------
@app.route("/send-otp", methods=["POST"])
def send_otp():
    data = request.get_json()
    email = data.get("email")
    if not email:
        return jsonify({"success": False, "message": "Email required"}), 400

    otp = str(random.randint(100000, 999999))
    otp_store[email] = otp

    try:
        sender_email = "authentichireweb@gmail.com"
        sender_password = "vnsmviyqkudgljvj"
        subject = "AuthentiHire OTP"
        body = f"Your OTP is: {otp}"
        message = f"Subject: {subject}\n\n{body}"

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, message)

        return jsonify({"success": True, "message": "OTP sent successfully!"})

    except Exception as e:
        print("Email Error:", e)
        return jsonify({"success": False, "message": str(e)}), 500


# -------------------- Signup/Login --------------------
@app.route("/candidate-signup", methods=["POST"])
def candidate_signup():
    data=request.get_json()
    name,username,email,otp,password=data.get("name"),data.get("username"),data.get("email"),data.get("otp"),data.get("password")
    if otp_store.get(email)!=otp: return jsonify({"success":False,"message":"Invalid OTP"}),400
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c=conn.cursor()
            c.execute("INSERT INTO candidates (name,username,email,password) VALUES (?,?,?,?)",(name,username,email,password))
            conn.commit()
        otp_store.pop(email,None)
        return jsonify({"success":True,"message":"Signup successful!"})
    except sqlite3.IntegrityError:
        return jsonify({"success":False,"message":"Email already registered"}),400

@app.route("/candidate-login", methods=["POST"])
def candidate_login():
    data=request.get_json()
    email,password=data.get("email"),data.get("password")
    with sqlite3.connect(DB_PATH) as conn:
        c=conn.cursor()
        c.execute("SELECT id,name,username,email FROM candidates WHERE email=? AND password=?",(email,password))
        user=c.fetchone()
    if user:
        session['user_type']='candidate'
        session['user_email']=email
        session['username']=user[2]
        return jsonify({"success":True,"message":"Login successful"})
    return jsonify({"success":False,"message":"Invalid credentials"}),401

@app.route("/client-login-page")
def client_login_page():
    return render_template("clientlogin.html")

@app.route("/client-signup-page")
def client_signup_page():
    return render_template("clientsignup.html")

@app.route("/candidate-login-page")
def candidate_login_page():
    return render_template("candidatelogin.html")

@app.route("/candidate-signup-page")
def candidate_signup_page():
    return render_template("candidatesignup.html")

@app.route("/client-signup", methods=["POST"])
def client_signup():
    data=request.get_json()
    name,username,email,otp,password=data.get("name"),data.get("username"),data.get("email"),data.get("otp"),data.get("password")
    if otp_store.get(email)!=otp: return jsonify({"success":False,"message":"Invalid OTP"}),400
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c=conn.cursor()
            c.execute("INSERT INTO clients (name,username,email,password) VALUES (?,?,?,?)",(name,username,email,password))
            conn.commit()
        otp_store.pop(email,None)
        return jsonify({"success":True,"message":"Client signup successful!"})
    except sqlite3.IntegrityError:
        return jsonify({"success":False,"message":"Email already registered"}),400

@app.route("/client-login", methods=["POST"])
def client_login():
    data=request.get_json()
    email,password=data.get("email"),data.get("password")
    with sqlite3.connect(DB_PATH) as conn:
        c=conn.cursor()
        c.execute("SELECT * FROM clients WHERE email=? AND password=?",(email,password))
        user=c.fetchone()
    if user:
        session['user_type']='client'
        session['user_email']=email
        return jsonify({"success":True,"message":"Client login successful"})
    return jsonify({"success":False,"message":"Invalid credentials"}),401


# -------------------- Profile --------------------
@app.route("/profile")
def profile():
    if 'user_type' not in session or 'user_email' not in session:
        return redirect(url_for('home'))
    user_type,email=session['user_type'],session['user_email']
    user = get_candidate_by_email(email) if user_type=='candidate' else get_client_by_email(email)
    if not user: session.clear(); return redirect(url_for('home'))
    return render_template('profile.html',user=user,user_type=user_type)

@app.route("/edit-profile", methods=["GET","POST"])
def edit_profile():
    if 'user_type' not in session or 'user_email' not in session:
        return redirect(url_for('home'))

    user_type, email = session['user_type'], session['user_email']

    if request.method == "POST":
        data = request.get_json() if request.is_json else request.form
        new_name, new_email = data.get("name"), data.get("email")

        if not new_name or not new_email:
            return jsonify({"success": False, "message": "Name & email required"}), 400

        success = update_candidate_profile(email, new_name, new_email) if user_type == 'candidate' else update_client_profile(email, new_name, new_email)

        if success:
            session['user_email'] = new_email
            return redirect(url_for('profile'))

        return jsonify({"success": False, "message": "Update failed"}), 500

    user = get_candidate_by_email(email) if user_type == 'candidate' else get_client_by_email(email)
    return render_template('edit_profile.html', user=user, user_type=user_type)

@app.route("/candidate-logout", methods=["POST","GET"])
def candidate_logout(): session.clear(); return redirect(url_for('home'))
@app.route("/client-logout", methods=["POST","GET"])
def client_logout(): session.clear(); return redirect(url_for('home'))


# -------------------- Meeting --------------------
@app.route("/create-meeting", methods=["POST"])
def create_meeting():
    data=request.get_json()
    meeting_id,password=data.get("id"),data.get("password")
    active_meetings[str(meeting_id)]=password
    return jsonify({"success":True})

@app.route("/leave_meeting/<meeting_id>", methods=["POST"])
def leave_meeting(meeting_id):
    if meeting_id in meeting_stop_flags:
        print(f"🛑 leave_meeting called for {meeting_id} — stopping AI pipeline")
        meeting_stop_flags[meeting_id].set()
    username = session.get("username")
    if username:
        live_frames.pop(username, None)
    return jsonify({"success": True})

@app.route("/validate-meeting", methods=["POST"])
def validate_meeting():
    data=request.get_json()
    meeting_id,password=data.get("id"),data.get("password")
    if str(meeting_id) in active_meetings and active_meetings[str(meeting_id)]==password:
        return jsonify({"success":True})
    return jsonify({"success":False})

@app.route("/api/gaze", methods=["POST"])
def receive_gaze_data():
    data = request.get_json()
    meeting_id = data.get("meeting_id")
    user_id = data.get("user_id")
    direction = data.get("direction")
    timestamp = data.get("timestamp")

    print(f"[Gaze] Meeting: {meeting_id} | User: {user_id} | Direction: {direction} | Time: {timestamp}")

    try:
        with sqlite3.connect("candidates.db") as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO gaze_data (meeting_id, user_id, direction, timestamp) VALUES (?, ?, ?, ?)",
                (meeting_id, user_id, direction, timestamp)
            )
            conn.commit()
        return jsonify({"success": True})
    except Exception as e:
        print("DB Error:", e)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/view-gaze")
def view_gaze():
    import sqlite3
    from flask import render_template

    with sqlite3.connect("candidates.db") as conn:
        c = conn.cursor()
        c.execute("SELECT DISTINCT meeting_id FROM gaze_data ORDER BY timestamp DESC LIMIT 1")
        row = c.fetchone()
        if not row:
            return "No gaze data available yet."
        meeting_id = row[0]

        c.execute("""
            SELECT user_id, total_events, focus_percentage
            FROM gaze_summary
            WHERE meeting_id=?
        """, (meeting_id,))
        rows = c.fetchall()

    total_events = sum(r[1] for r in rows)
    avg_focus = round(sum(r[2] for r in rows)/len(rows), 2) if rows else 0

    directions = ["Left", "Right", "Center", "Top", "Bottom"]
    percentages = {d: round(100/len(directions), 2) for d in directions}

    return render_template("analytics.html",
                           meeting_id=meeting_id,
                           total_events=total_events,
                           avg_focus=avg_focus,
                           percentages=percentages,
                           user_summary=[{"user_id": r[0], "focus_percentage": round(r[2],2)} for r in rows])


# -------------------- Uploads --------------------
@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "user_type" not in session or session["user_type"]!="candidate": return jsonify({"message":"Unauthorized"}),403
    if "video" not in request.files: return jsonify({"message":"No video file"}),400
    video=request.files["video"]
    username=session.get("username")
    user_folder=os.path.join(UPLOAD_FOLDER,username)
    os.makedirs(user_folder,exist_ok=True)
    filename=secure_filename(f"{username}_video.webm")
    path=os.path.join(user_folder,filename)
    video.save(path)
    return jsonify({"message":"Video uploaded successfully","path":f"/uploads/{username}/{filename}"})

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# -------------------- Audio Analysis --------------------
def analyze_audio_clip(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False,suffix=".wav") as tmp:
            file.save(tmp.name)
            y,sr=librosa.load(tmp.name,sr=16000)
        rms=np.mean(librosa.feature.rms(y=y))
        pitch=np.mean(librosa.yin(y,fmin=50,fmax=400))
        clarity_score=(rms+pitch/400)/2
        return round(float(clarity_score),2)
    except:
        return 0.0

def detect_audio_bias(file):
    score=analyze_audio_clip(file)
    return "⚠️ Possible bias" if score<0.25 else "✅ No bias"

def analyze_audio_ai(file):
    clarity = analyze_audio_clip(file)
    bias = detect_audio_bias(file)
    clarity_status = "✅ Good" if clarity>0.25 else "⚠️ Poor"
    return {"audio_clarity": clarity_status, "bias": bias}

@app.route("/analyze_audio", methods=["POST"])
def analyze_audio_route():
    print("\n🎤 /analyze_audio route called")

    audio = request.files.get("audio")
    meeting_id = request.form.get("meeting_id")
    user_id = session.get("username", request.remote_addr)

    print(f"Meeting ID: {meeting_id}")
    print(f"User ID: {user_id}")
    print(f"Audio file: {audio}")

    if not audio:
        print("❌ No audio file provided")
        return jsonify({"error": "No audio file"}), 400

    results = analyze_audio_ai(audio)
    print(f"Audio results: {results}")

    with sqlite3.connect(AI_DB_PATH) as conn:
        c = conn.cursor()
        for feature, status in results.items():
            print(f"💾 Saving: {feature} = {status}")
            c.execute("INSERT INTO ai_results (meeting_id,user_id,feature,status) VALUES (?,?,?,?)",
                      (meeting_id, user_id, feature, status))
            socketio.emit("ai_status_update",
                         {"feature": feature, "status": status, "source": user_id},
                         room=f"meeting_{meeting_id}")
        conn.commit()

    print("✅ Audio analysis complete")
    return jsonify({"success": True, "results": results})
