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
