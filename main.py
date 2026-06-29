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

@app.route("/")
def home():
    return "AI Attendance System - Backend Running"

if __name__ == "__main__":
    print("Server running on http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)