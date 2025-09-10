import streamlit as st
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import firebase_admin
from firebase_admin import credentials, db
import json
import datetime
import tempfile

# ------------------ FIREBASE SETUP ------------------
if "firebase" in st.secrets:  # safer (Streamlit Cloud secrets)
    cred = credentials.Certificate(dict(st.secrets["firebase"]))
else:  # fallback (local JSON file, not for deployment)
    cred = credentials.Certificate("firebase_key.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://console.firebase.google.com/u/0/project/smart-attendance-system-45296/database/smart-attendance-system-45296-default-rtdb/data/~2F/"
    })

# Firebase references
students_ref = db.reference("students")
attendance_ref = db.reference("attendance")

# ------------------ FACE RECOGNITION SETUP ------------------
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# ------------------ DATABASE HELPERS ------------------
def save_student_profile(name, embeddings):
    """Save multiple embeddings for one student."""
    students_ref.child(name).set([e.tolist() for e in embeddings])

def load_student_profiles():
    """Load all student embeddings."""
    data = students_ref.get()
    if not data:
        return {}
    return {k: [np.array(e) for e in v] for k, v in data.items()}

def save_attendance(date, log):
    attendance_ref.child(date).set(log)

def load_attendance():
    data = attendance_ref.get()
    return data if data else {}

# ------------------ ATTENDANCE FUNCTIONS ------------------
def register_students():
    """Register students with 3 images each."""
    st.subheader("📸 Register Students")

    num_students = st.number_input("Enter total number of students:", min_value=1, step=1)

    student_db = load_student_profiles()

    for i in range(num_students):
        name = st.text_input(f"Enter name for Student {i+1}")
        uploads = st.file_uploader(
            f"Upload 3 images for {name}", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f"u{i}"
        )

        if st.button(f"Save {name}"):
            if name and len(uploads) >= 3:
                embeddings = []
                for file in uploads[:3]:
                    temp = tempfile.NamedTemporaryFile(delete=False)
                    temp.write(file.read())
                    img = cv2.imread(temp.name)
                    faces = app.get(img)
                    if faces:
                        embeddings.append(faces[0].embedding)
                if embeddings:
                    save_student_profile(name, embeddings)
                    st.success(f"{name} registered successfully ✅")
            else:
                st.error("Please provide a name and at least 3 images!")

def take_attendance():
    """Mark attendance from a video."""
    st.subheader("🎥 Take Attendance")

    date = st.date_input("Select Date", value=datetime.date.today())
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if video_file and st.button("Process Video"):
        student_db = load_student_profiles()
        attendance_log = load_attendance()

        if str(date) not in attendance_log:
            attendance_log[str(date)] = {}

        present_today = set(attendance_log[str(date)].keys())

        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(video_file.read())
        cap = cv2.VideoCapture(temp.name)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("annotated.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        threshold = 0.5
        present_students = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            faces = app.get(frame)
            for face in faces:
                emb = face.embedding
                best_match, best_score = None, -1
                for student, embeds in student_db.items():
                    for ref_emb in embeds:
                        sim = np.dot(emb, ref_emb) / (np.linalg.norm(emb) * np.linalg.norm(ref_emb))
                        if sim > best_score:
                            best_score, best_match = sim, student

                if best_match and best_score > threshold:
                    if best_match not in present_today:
                        attendance_log[str(date)][best_match] = {"status": "Present", "confidence": float(best_score)}
                        present_today.add(best_match)
                    cv2.rectangle(frame, (face.bbox[0], face.bbox[1]), (face.bbox[2], face.bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"{best_match} {best_score:.2f}", (face.bbox[0], face.bbox[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            out.write(frame)

        cap.release()
        out.release()
        save_attendance(str(date), attendance_log[str(date)])

        st.success("Attendance marked and video annotated ✅")
        st.video("annotated.mp4")

        present = list(attendance_log[str(date)].keys())
        absent = [s for s in student_db.keys() if s not in present]

        st.write("### ✅ Present Students")
        st.write(present)

        st.write("### ❌ Absent Students")
        st.write(absent)

def view_students():
    """Show list of registered students."""
    st.subheader("👨‍🎓 Registered Students")
    student_db = load_student_profiles()
    if student_db:
        st.write(list(student_db.keys()))
    else:
        st.warning("No students registered yet.")

def view_attendance():
    """Show attendance logs."""
    st.subheader("📊 Attendance Log")
    attendance_log = load_attendance()
    if attendance_log:
        st.json(attendance_log)
    else:
        st.warning("No attendance records yet.")

# ------------------ MAIN MENU ------------------
def main():
    st.title("🎓 Smart Attendance System (Face Recognition + Firebase)")

    menu = ["Register Students", "Take Attendance", "View Students", "View Attendance Logs", "Exit"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Register Students":
        register_students()
    elif choice == "Take Attendance":
        take_attendance()
    elif choice == "View Students":
        view_students()
    elif choice == "View Attendance Logs":
        view_attendance()
    elif choice == "Exit":
        st.write("👋 Thank you for using the Smart Attendance System!")

if __name__ == "__main__":
    main()

