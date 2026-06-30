import os
import sqlite3
from flask import Flask, jsonify, request

app = Flask(__name__)

# Database setup
DB = "attendance.db"

def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS students(
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        roll_no TEXT UNIQUE,
        name    TEXT,
        course  TEXT
    );
    CREATE TABLE IF NOT EXISTS subjects(
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_name TEXT UNIQUE
    );
    CREATE TABLE IF NOT EXISTS enrollments(
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        subject_id INTEGER,
        UNIQUE(student_id, subject_id)
    );
    CREATE TABLE IF NOT EXISTS attendance(
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        subject_id INTEGER,
        date       TEXT,
        time       TEXT,
        UNIQUE(student_id, subject_id, date)
    );
    """)
    conn.commit()
    conn.close()

init_db()

# Student Routes
@app.route("/api/student/register", methods=["POST"])
def register_student():
    data   = request.json or {}
    roll   = data.get("roll_no","").strip()
    name   = data.get("name","").strip()
    course = data.get("course","").strip()
    if not roll or not name or not course:
        return jsonify({"success":False,"message":"All fields required!"}), 400
    conn = get_db()
    try:
        if conn.execute("SELECT id FROM students WHERE roll_no=?",(roll,)).fetchone():
            return jsonify({"success":False,"message":f"{roll} already registered!"})
        conn.execute("INSERT INTO students(roll_no,name,course) VALUES(?,?,?)",(roll,name,course))
        conn.commit()
        return jsonify({"success":True,"message":f"{name} ({roll}) registered!"})
    finally:
        conn.close()

@app.route("/api/students")
def list_students():
    conn = get_db()
    try:
        rows = conn.execute("SELECT * FROM students ORDER BY name").fetchall()
        return jsonify({"success":True,"students":[dict(r) for r in rows]})
    finally:
        conn.close()
# Subject Routes
@app.route("/api/subject/add", methods=["POST"])
def add_subject():
    data = request.json or {}
    name = data.get("subject_name","").strip()
    if not name:
        return jsonify({"success":False,"message":"Subject name required!"}), 400
    conn = get_db()
    try:
        if conn.execute("SELECT id FROM subjects WHERE subject_name=?",(name,)).fetchone():
            return jsonify({"success":False,"message":f"'{name}' already exists!"})
        conn.execute("INSERT INTO subjects(subject_name) VALUES(?)",(name,))
        conn.commit()
        return jsonify({"success":True,"message":f"Subject '{name}' added!"})
    finally:
        conn.close()

def list_subjects():
    conn = get_db()
    try:
        rows = conn.execute("SELECT * FROM subjects ORDER BY subject_name").fetchall()
        return jsonify({"success":True,"subjects":[dict(r) for r in rows]})
    finally:
        conn.close()

# Enrollment Routes
@app.route("/api/enroll", methods=["POST"])
def enroll_student():
    # Enrollment Routes
@app.route("/api/enroll", methods=["POST"])
def enroll_student():
    data = request.json or {}
    roll = data.get("roll_no","").strip()
    sid  = data.get("subject_id")
    if not roll or not sid:
        return jsonify({"success":False,"message":"roll_no aur subject_id required!"}), 400
    conn = get_db()
    try:
        student = conn.execute("SELECT id FROM students WHERE roll_no=?",(roll,)).fetchone()
        if not student:
            return jsonify({"success":False,"message":f"{roll} nahi mila!"})
        subject = conn.execute("SELECT id,subject_name FROM subjects WHERE id=?",(sid,)).fetchone()
        if not subject:
            return jsonify({"success":False,"message":"Subject nahi mila!"})
        if conn.execute("SELECT id FROM enrollments WHERE student_id=? AND subject_id=?",(student["id"],sid)).fetchone():
            return jsonify({"success":False,"message":f"{roll} already enrolled!"})
        conn.execute("INSERT INTO enrollments(student_id,subject_id) VALUES(?,?)",(student["id"],sid))
        conn.commit()
        return jsonify({"success":True,"message":f"{roll} enrolled in '{subject['subject_name']}'"})
    finally:
        conn.close()

@app.route("/api/enrollments/<int:subject_id>")
def list_enrollments(subject_id):
    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT s.id,s.roll_no,s.name,s.course
            FROM students s JOIN enrollments e ON e.student_id=s.id
            WHERE e.subject_id=? ORDER BY s.name
        """,(subject_id,)).fetchall()
        return jsonify({"success":True,"students":[dict(r) for r in rows]})
    finally:
        conn.close()

@app.route("/")
def home():
    return "AI Attendance System - Backend Running"

if __name__ == "__main__":
    print("Server running on http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)