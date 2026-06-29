import os
import cv2
import sqlite3
import numpy as np
import csv
import threading
import queue
from datetime import date, datetime
from flask import Flask, jsonify, request, render_template_string

app = Flask(__name__)

# ─────────────────────────────────────────
# CAMERA TASK QUEUE
# OpenCV window sirf main thread pe dikhti hai Windows mein.
# Isliye sab camera kaam main thread karta hai via queue.
# ─────────────────────────────────────────
_cam_queue = queue.Queue()

def run_camera_task(func):
    """
    Flask worker thread se main thread ko camera task bhejo.
    Block karta hai jab tak result na aaye.
    """
    result_q = queue.Queue()
    _cam_queue.put((func, result_q))
    return result_q.get(timeout=180)   # 3 min max wait

def process_camera_queue():
    """
    Main thread mein ye loop chalta hai.
    Queue se tasks uthata hai aur execute karta hai.
    Flask se alag — sirf camera kaam karta hai.
    """
    while True:
        try:
            func, result_q = _cam_queue.get(timeout=0.1)
            try:
                result = func()
                result_q.put(result)
            except Exception as e:
                result_q.put({"success": False, "message": str(e)})
        except queue.Empty:
            pass

# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────
os.makedirs("faces", exist_ok=True)
DB = "attendance.db"

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ─────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────
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

# ─────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>AI Attendance System</title>
<style>
  :root {
    --bg:      #f0f7ff;
    --panel:   #ffffff;
    --card:    #ffffff;
    --border:  #d0e4f7;
    --accent:  #2563eb;
    --green:   #059669;
    --red:     #dc2626;
    --yellow:  #d97706;
    --text:    #1e293b;
    --sub:     #64748b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; display: flex; min-height: 100vh; }

  #sidebar {
    width: 220px; background: #1e3a5f; border-right: 1px solid var(--border);
    display: flex; flex-direction: column; position: fixed;
    top: 0; left: 0; bottom: 0; z-index: 100; overflow-y: auto;
  }
  .brand {
    padding: 20px 18px; border-bottom: 1px solid rgba(255,255,255,0.1);
    display: flex; align-items: center; gap: 10px;
  }
  .brand-icon { width: 36px; height: 36px; background: #3b82f6; border-radius: 9px; display: flex; align-items: center; justify-content: center; font-size: 18px; }
  .brand-name { font-size: 15px; font-weight: 700; color: #ffffff; }
  .brand-sub  { font-size: 10px; color: rgba(255,255,255,0.5); }
  .nav-label  { font-size: 10px; text-transform: uppercase; letter-spacing:.08em; color: rgba(255,255,255,0.4); padding: 14px 18px 4px; font-weight: 600; }
  .nav-link   { display: flex; align-items: center; gap: 10px; padding: 9px 18px; color: rgba(255,255,255,0.65); text-decoration: none; font-size: 13px; border-left: 3px solid transparent; cursor: pointer; transition: all .15s; }
  .nav-link:hover  { color: #ffffff; background: rgba(255,255,255,0.08); }
  .nav-link.active { color: #7dd3fc; background: rgba(59,130,246,0.2); border-left-color: #3b82f6; font-weight: 600; }

  #main { margin-left: 220px; flex: 1; padding: 28px; }
  .page { display: none; }
  .page.active { display: block; }
  .page-title { font-size: 22px; font-weight: 700; margin-bottom: 6px; }
  .page-sub   { font-size: 13px; color: var(--sub); margin-bottom: 24px; }

  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 22px; margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
  .card-title { font-size: 11px; font-weight: 600; margin-bottom: 16px; color: var(--sub); text-transform: uppercase; letter-spacing: .05em; }

  .form-row   { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }
  .form-group { display: flex; flex-direction: column; gap: 5px; flex: 1; min-width: 180px; }
  label { font-size: 11px; color: var(--sub); font-weight: 600; }
  input, select { background: #f8fafc; border: 1px solid var(--border); color: var(--text); padding: 9px 12px; border-radius: 8px; font-size: 13px; outline: none; transition: border-color .15s; }
  input:focus, select:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(37,99,235,0.1); }
  select option { background: #ffffff; color: var(--text); }

  .btn { padding: 9px 20px; border-radius: 8px; border: none; font-size: 13px; font-weight: 600; cursor: pointer; transition: all .15s; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-primary:hover { background: #1d4ed8; }
  .btn-success { background: var(--green); color: #fff; }
  .btn-success:hover { background: #047857; }
  .btn-ghost   { background: transparent; border: 1px solid var(--border); color: var(--sub); }
  .btn-ghost:hover { border-color: var(--accent); color: var(--accent); background: #eff6ff; }
  .btn:disabled { opacity: .5; cursor: not-allowed; }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 10px 14px; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: var(--sub); border-bottom: 1px solid var(--border); background: #f1f5f9; }
  td { padding: 11px 14px; border-bottom: 1px solid var(--border); vertical-align: middle; }
  tr:hover td { background: #eff6ff; }
  tr:last-child td { border-bottom: none; }

  .badge { display: inline-block; padding: 3px 10px; border-radius: 99px; font-size: 11px; font-weight: 600; }
  .badge-purple { background: #ede9fe; color: #5b21b6; }
  .badge-green  { background: #d1fae5; color: #065f46; }
  .badge-red    { background: #fee2e2; color: #991b1b; }

  #toast { position: fixed; bottom: 24px; right: 24px; padding: 12px 20px; border-radius: 10px; font-size: 13px; font-weight: 600; display: none; z-index: 999; max-width: 320px; border: 1px solid; }
  #toast.success { background: #d1fae5; color: #065f46; border-color: #6ee7b7; }
  #toast.error   { background: #fee2e2; color: #991b1b; border-color: #fca5a5; }
  #toast.info    { background: #dbeafe; color: #1e40af; border-color: #93c5fd; }

  .stats-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 20px; }
  .stat-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  .stat-val  { font-size: 28px; font-weight: 700; color: var(--accent); }
  .stat-lbl  { font-size: 12px; color: var(--sub); margin-top: 4px; }

  .log-box { background: #1e293b; border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; font-family: Consolas, monospace; font-size: 12px; height: 200px; overflow-y: auto; margin-top: 14px; }
  .log-ok   { color: #4ade80; }
  .log-err  { color: #f87171; }
  .log-info { color: #7dd3fc; }
  .log-warn { color: #fbbf24; }

  .empty { text-align: center; padding: 40px; color: var(--sub); font-size: 13px; }
  code { background: #eff6ff; color: #1d4ed8; padding: 2px 7px; border-radius: 4px; font-size: 12px; }

  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid rgba(37,99,235,.3); border-top-color: #2563eb; border-radius: 50%; animation: spin .7s linear infinite; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .notice { background: #fefce8; border: 1px solid #fde047; border-radius: 8px; padding: 10px 14px; font-size: 12px; color: #854d0e; margin-bottom: 14px; }
</style>
</head>
<body>

<nav id="sidebar">
  <div class="brand">
    <div class="brand-icon">👁</div>
    <div><div class="brand-name">AttendAI</div><div class="brand-sub">Face Recognition</div></div>
  </div>
  <div class="nav-label">Main</div>
  <div class="nav-link active" onclick="showPage('dashboard',this)">🏠 Dashboard</div>
  <div class="nav-label">Management</div>
  <div class="nav-link" onclick="showPage('students',this)">👤 Students</div>
  <div class="nav-link" onclick="showPage('subjects',this)">📚 Subjects</div>
  <div class="nav-link" onclick="showPage('enroll',this)">📋 Enrollment</div>
  <div class="nav-label">Attendance</div>
  <div class="nav-link" onclick="showPage('capture',this)">📸 Capture Face</div>
  <div class="nav-link" onclick="showPage('train',this)">🧠 Train Model</div>
  <div class="nav-link" onclick="showPage('attendance',this)">✅ Take Attendance</div>
  <div class="nav-label">Reports</div>
  <div class="nav-link" onclick="showPage('report',this)">📊 View Report</div>
</nav>

<div id="main">

  <!-- DASHBOARD -->
  <div class="page active" id="page-dashboard">
    <div class="page-title">Dashboard</div>
    <div class="page-sub">System overview</div>
    <div class="stats-row">
      <div class="stat-card"><div class="stat-val" id="st-students">—</div><div class="stat-lbl">Students</div></div>
      <div class="stat-card"><div class="stat-val" id="st-subjects">—</div><div class="stat-lbl">Subjects</div></div>
      <div class="stat-card"><div class="stat-val" id="st-today">—</div><div class="stat-lbl">Today's Attendance</div></div>
      <div class="stat-card"><div class="stat-val" id="st-total">—</div><div class="stat-lbl">Total Records</div></div>
    </div>
    <div class="card">
      <div class="card-title">Quick Actions</div>
      <div style="display:flex;gap:10px;flex-wrap:wrap">
        <button class="btn btn-primary" onclick="showPage('students',null)">👤 Register Student</button>
        <button class="btn btn-ghost"   onclick="showPage('capture',null)">📸 Capture Face</button>
        <button class="btn btn-ghost"   onclick="showPage('train',null)">🧠 Train Model</button>
        <button class="btn btn-success" onclick="showPage('attendance',null)">✅ Take Attendance</button>
      </div>
    </div>
  </div>

  <!-- STUDENTS -->
  <div class="page" id="page-students">
    <div class="page-title">Students</div>
    <div class="page-sub">Register and manage students</div>
    <div class="card">
      <div class="card-title">Register New Student</div>
      <div class="form-row">
        <div class="form-group"><label>Roll No</label><input id="s-roll" placeholder="F2024000001"/></div>
        <div class="form-group"><label>Full Name</label><input id="s-name" placeholder="Ali Hassan"/></div>
        <div class="form-group"><label>Course</label><input id="s-course" placeholder="BS Cyber Security"/></div>
      </div>
      <button class="btn btn-primary" onclick="registerStudent()">Register Student</button>
    </div>
    <div class="card">
      <div class="card-title">All Students</div>
      <table><thead><tr><th>Roll No</th><th>Name</th><th>Course</th></tr></thead>
      <tbody id="students-table"><tr><td colspan="3" class="empty">Loading...</td></tr></tbody></table>
    </div>
  </div>

  <!-- SUBJECTS -->
  <div class="page" id="page-subjects">
    <div class="page-title">Subjects</div>
    <div class="page-sub">Add and manage subjects</div>
    <div class="card">
      <div class="card-title">Add New Subject</div>
      <div class="form-row">
        <div class="form-group"><label>Subject Name</label><input id="sub-name" placeholder="Information Assurance"/></div>
      </div>
      <button class="btn btn-primary" onclick="addSubject()">Add Subject</button>
    </div>
    <div class="card">
      <div class="card-title">All Subjects</div>
      <table><thead><tr><th>ID</th><th>Subject Name</th></tr></thead>
      <tbody id="subjects-table"><tr><td colspan="2" class="empty">Loading...</td></tr></tbody></table>
    </div>
  </div>

  <!-- ENROLLMENT -->
  <div class="page" id="page-enroll">
    <div class="page-title">Enrollment</div>
    <div class="page-sub">Enroll students in subjects</div>
    <div class="card">
      <div class="card-title">Enroll Student</div>
      <div class="form-row">
        <div class="form-group"><label>Student</label><select id="e-student"><option value="">Select student...</option></select></div>
        <div class="form-group"><label>Subject</label><select id="e-subject"><option value="">Select subject...</option></select></div>
      </div>
      <button class="btn btn-primary" onclick="enrollStudent()">Enroll</button>
    </div>
    <div class="card">
      <div class="card-title">View Enrolled Students</div>
      <div class="form-row">
        <div class="form-group"><label>Select Subject</label><select id="e-view-subject" onchange="loadEnrollments()"><option value="">Select subject...</option></select></div>
      </div>
      <table><thead><tr><th>Roll No</th><th>Name</th><th>Course</th></tr></thead>
      <tbody id="enroll-table"><tr><td colspan="3" class="empty">Subject select karo upar</td></tr></tbody></table>
    </div>
  </div>

  <!-- CAPTURE FACE -->
  <div class="page" id="page-capture">
    <div class="page-title">Capture Face</div>
    <div class="page-sub">30 photos lete hain face recognition ke liye</div>
    <div class="card">
      <div class="notice">⚠️ Camera window alag se khulegi — taskbar mein dekho agar screen pe nahi dikh rahi!</div>
      <div class="card-title">Student Select Karo</div>
      <div class="form-row">
        <div class="form-group"><label>Student</label><select id="cap-student"><option value="">Select student...</option></select></div>
      </div>
      <button class="btn btn-primary" id="btn-capture" onclick="captureface()">📸 Start Capture</button>
      <div class="log-box" id="cap-log"><div class="log-info">[--:--:--] Student select karo aur Start Capture dabao.</div></div>
    </div>
  </div>

  <!-- TRAIN MODEL -->
  <div class="page" id="page-train">
    <div class="page-title">Train Model</div>
    <div class="page-sub">Captured images se face recognition model train karo</div>
    <div class="card">
      <p style="font-size:13px;color:var(--sub);margin-bottom:16px">Pehle sab students ke faces capture karo, phir train karo.</p>
      <button class="btn btn-primary" id="btn-train" onclick="trainModel()">🧠 Start Training</button>
      <div class="log-box" id="train-log"><div class="log-info">[--:--:--] Start Training dabao.</div></div>
    </div>
  </div>

  <!-- ATTENDANCE -->
  <div class="page" id="page-attendance">
    <div class="page-title">Take Attendance</div>
    <div class="page-sub">Roll no. type karo — face scan + blink verification hogi</div>
    <div class="card">
      <div class="notice">⚠️ Camera window alag se khulegi — taskbar mein dekho!</div>
      <div class="form-row">
        <div class="form-group"><label>Subject Select Karo</label><select id="att-subject"><option value="">Select subject...</option></select></div>
        <div class="form-group"><label>Student Roll No</label><input id="att-roll" placeholder="F2024000001" onkeydown="if(event.key==='Enter') markOne()"/></div>
      </div>
      <div style="display:flex;gap:10px">
        <button class="btn btn-success" id="btn-mark" onclick="markOne()">✅ Scan & Mark</button>
        <button class="btn btn-ghost" onclick="document.getElementById('att-log').innerHTML=''">Clear Log</button>
      </div>
      <div class="log-box" id="att-log"><div class="log-info">[--:--:--] Subject select karo, roll no. likho, Enter dabao.</div></div>
    </div>
    <div class="card">
      <div class="card-title">Aaj Mark Hue</div>
      <div id="marked-list" style="display:flex;flex-wrap:wrap;gap:8px;min-height:36px">
        <span style="color:var(--sub);font-size:12px">Koi nahi abhi tak...</span>
      </div>
    </div>
  </div>

  <!-- REPORT -->
  <div class="page" id="page-report">
    <div class="page-title">Attendance Report</div>
    <div class="page-sub">Poori attendance records</div>
    <div class="card">
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
        <select id="rep-subject" style="flex:1;max-width:280px"><option value="">All Subjects</option></select>
        <button class="btn btn-primary" onclick="loadReport()">Load</button>
        <button class="btn btn-ghost"   onclick="exportCSV()">⬇ Export CSV</button>
      </div>
    </div>
    <div class="card">
      <table><thead><tr><th>Roll No</th><th>Name</th><th>Subject</th><th>Date</th><th>Time</th></tr></thead>
      <tbody id="report-table"><tr><td colspan="5" class="empty">Load dabao</td></tr></tbody></table>
    </div>
  </div>

</div>
<div id="toast"></div>

<script>
function ts() { return new Date().toLocaleTimeString('en-US',{hour12:false}); }
function logLine(boxId, msg, cls='') {
  const box = document.getElementById(boxId);
  const d = document.createElement('div');
  d.className = cls ? 'log-'+cls : '';
  d.textContent = '['+ts()+'] '+msg;
  box.appendChild(d);
  box.scrollTop = box.scrollHeight;
}
function toast(msg, type='success') {
  const t = document.getElementById('toast');
  t.textContent = msg; t.className = type; t.style.display = 'block';
  setTimeout(() => t.style.display='none', 3500);
}
async function api(url, method='GET', body=null) {
  const opts = {method, headers:{'Content-Type':'application/json'}};
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(url, opts);
  return r.json();
}

function showPage(name, el) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  document.getElementById('page-'+name).classList.add('active');
  if (el) el.classList.add('active');
  if (name==='dashboard')  loadDashboard();
  if (name==='students')   loadStudents();
  if (name==='subjects')   loadSubjects();
  if (name==='enroll')     loadEnrollPage();
  if (name==='capture')    loadCaptureDropdown();
  if (name==='attendance') loadAttSubjects();
  if (name==='report')     loadRepSubjects();
}

async function loadDashboard() {
  const [s, sub, rep] = await Promise.all([api('/api/students'), api('/api/subjects'), api('/api/attendance/report')]);
  document.getElementById('st-students').textContent = s.students?.length || 0;
  document.getElementById('st-subjects').textContent = sub.subjects?.length || 0;
  document.getElementById('st-total').textContent    = rep.total || 0;
  const today = new Date().toISOString().split('T')[0];
  document.getElementById('st-today').textContent = (rep.records||[]).filter(r=>r.date===today).length;
}

async function loadStudents() {
  const d = await api('/api/students');
  const tb = document.getElementById('students-table');
  if (!d.students?.length) { tb.innerHTML='<tr><td colspan="3"><div class="empty">Koi student nahi.</div></td></tr>'; return; }
  tb.innerHTML = d.students.map(s=>`<tr><td><code>${s.roll_no}</code></td><td>${s.name}</td><td><span class="badge badge-purple">${s.course}</span></td></tr>`).join('');
}
async function registerStudent() {
  const roll=document.getElementById('s-roll').value.trim();
  const name=document.getElementById('s-name').value.trim();
  const course=document.getElementById('s-course').value.trim();
  if (!roll||!name||!course) { toast('Sab fields bharo!','error'); return; }
  const d = await api('/api/student/register','POST',{roll_no:roll,name,course});
  toast(d.message, d.success?'success':'error');
  if (d.success) { document.getElementById('s-roll').value=''; document.getElementById('s-name').value=''; document.getElementById('s-course').value=''; loadStudents(); }
}

async function loadSubjects() {
  const d = await api('/api/subjects');
  const tb = document.getElementById('subjects-table');
  if (!d.subjects?.length) { tb.innerHTML='<tr><td colspan="2"><div class="empty">Koi subject nahi.</div></td></tr>'; return; }
  tb.innerHTML = d.subjects.map(s=>`<tr><td><code>${s.id}</code></td><td>${s.subject_name}</td></tr>`).join('');
}
async function addSubject() {
  const name = document.getElementById('sub-name').value.trim();
  if (!name) { toast('Subject name daalo!','error'); return; }
  const d = await api('/api/subject/add','POST',{subject_name:name});
  toast(d.message, d.success?'success':'error');
  if (d.success) { document.getElementById('sub-name').value=''; loadSubjects(); }
}

async function loadEnrollPage() {
  const [s,sub] = await Promise.all([api('/api/students'),api('/api/subjects')]);
  document.getElementById('e-student').innerHTML = '<option value="">Select student...</option>' + (s.students||[]).map(x=>`<option value="${x.roll_no}">${x.roll_no} — ${x.name}</option>`).join('');
  ['e-subject','e-view-subject'].forEach(id => {
    document.getElementById(id).innerHTML = '<option value="">Select subject...</option>' + (sub.subjects||[]).map(x=>`<option value="${x.id}">${x.subject_name}</option>`).join('');
  });
}
async function enrollStudent() {
  const roll=document.getElementById('e-student').value;
  const sid=document.getElementById('e-subject').value;
  if (!roll||!sid) { toast('Student aur subject dono select karo!','error'); return; }
  const d = await api('/api/enroll','POST',{roll_no:roll,subject_id:parseInt(sid)});
  toast(d.message, d.success?'success':'error');
  if (d.success) loadEnrollments();
}
async function loadEnrollments() {
  const sid = document.getElementById('e-view-subject').value;
  if (!sid) return;
  const d = await api(`/api/enrollments/${sid}`);
  const tb = document.getElementById('enroll-table');
  if (!d.students?.length) { tb.innerHTML='<tr><td colspan="3"><div class="empty">Koi enrolled nahi.</div></td></tr>'; return; }
  tb.innerHTML = d.students.map(s=>`<tr><td><code>${s.roll_no}</code></td><td>${s.name}</td><td>${s.course}</td></tr>`).join('');
}

async function loadCaptureDropdown() {
  const d = await api('/api/students');
  document.getElementById('cap-student').innerHTML = '<option value="">Select student...</option>' + (d.students||[]).map(s=>`<option value="${s.roll_no}">${s.roll_no} — ${s.name}</option>`).join('');
}
async function captureface() {
  const roll = document.getElementById('cap-student').value;
  if (!roll) { toast('Student select karo!','error'); return; }
  const btn = document.getElementById('btn-capture');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Camera chal rahi hai...';
  logLine('cap-log', `${roll} — camera khul rahi hai, taskbar check karo!`, 'info');
  logLine('cap-log', `Capturing 30 images... camera window mein face dikhao.`, 'info');
  const d = await api(`/api/face/capture/${roll}`);
  logLine('cap-log', d.message, d.success?'ok':'err');
  toast(d.message, d.success?'success':'error');
  btn.disabled = false; btn.innerHTML = '📸 Start Capture';
}

async function trainModel() {
  const btn = document.getElementById('btn-train');
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span>Training...';
  logLine('train-log', 'Model training shuru...', 'info');
  const d = await api('/api/face/train');
  logLine('train-log', d.message, d.success?'ok':'err');
  toast(d.message, d.success?'success':'error');
  btn.disabled = false; btn.innerHTML = '🧠 Start Training';
}

async function loadAttSubjects() {
  const d = await api('/api/subjects');
  document.getElementById('att-subject').innerHTML = '<option value="">Select subject...</option>' + (d.subjects||[]).map(s=>`<option value="${s.id}">${s.subject_name}</option>`).join('');
}
async function markOne() {
  const sid  = document.getElementById('att-subject').value;
  const roll = document.getElementById('att-roll').value.trim().toUpperCase();
  if (!sid)  { toast('Subject select karo!','error'); return; }
  if (!roll) { toast('Roll no. daalo!','error'); return; }
  const btn = document.getElementById('btn-mark');
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span>Scanning...';
  logLine('att-log', `${roll} — camera khul rahi hai, face dikhao...`, 'info');
  document.getElementById('att-roll').value = '';
  const d = await api('/api/attendance/single','POST',{roll_no:roll,subject_id:parseInt(sid)});
  btn.disabled = false; btn.innerHTML = '✅ Scan & Mark';
  const statusMap = {
    'marked':       ['ok',  `✅ ${d.name} (${roll}) — mark ho gai!`],
    'already_marked':['warn',`⚠️ ${roll} — aaj already mark tha!`],
    'not_found':    ['err', `❌ ${roll} — student nahi mila!`],
    'not_enrolled': ['err', `❌ ${roll} — is subject mein enrolled nahi!`],
    'face_mismatch':['err', `❌ ${roll} — face match nahi hua!`],
    'blink_failed': ['err', `❌ ${roll} — blink detect nahi hua!`],
  };
  const [cls, msg] = statusMap[d.status] || ['err', `❌ Error: ${d.message}`];
  logLine('att-log', msg, cls);
  toast(msg, d.status==='marked'?'success': d.status==='already_marked'?'info':'error');
  if (d.status==='marked') addMarkedBadge(roll, d.name);
}
function addMarkedBadge(roll, name) {
  const list = document.getElementById('marked-list');
  if (list.querySelector('span[style]')) list.innerHTML='';
  const b = document.createElement('span');
  b.className='badge badge-green'; b.textContent=`${roll} — ${name}`;
  list.appendChild(b);
}

async function loadRepSubjects() {
  const d = await api('/api/subjects');
  document.getElementById('rep-subject').innerHTML = '<option value="">All Subjects</option>' + (d.subjects||[]).map(s=>`<option value="${s.id}">${s.subject_name}</option>`).join('');
  loadReport();
}
async function loadReport() {
  const sid = document.getElementById('rep-subject').value;
  const url = sid ? `/api/attendance/report?subject_id=${sid}` : '/api/attendance/report';
  const d   = await api(url);
  const tb  = document.getElementById('report-table');
  if (!d.records?.length) { tb.innerHTML='<tr><td colspan="5"><div class="empty">Koi record nahi.</div></td></tr>'; return; }
  tb.innerHTML = d.records.map(r=>`<tr><td><code>${r.roll_no}</code></td><td>${r.name}</td><td><span class="badge badge-purple">${r.subject_name}</span></td><td>${r.date}</td><td style="color:var(--sub)">${r.time}</td></tr>`).join('');
}
function exportCSV() {
  const sid = document.getElementById('rep-subject').value;
  window.open(sid ? `/api/attendance/report?subject_id=${sid}` : '/api/attendance/report');
  toast('CSV export ho raha hai!','info');
}

loadDashboard();
</script>
</body>
</html>
"""

# ─────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────
@app.route("/")
def home():
    return render_template_string(HTML)

# ─────────────────────────────────────────
# STUDENT ROUTES
# ─────────────────────────────────────────
@app.route("/api/student/register", methods=["POST"])
def register_student():
    data   = request.json or {}
    roll   = data.get("roll_no","").strip()
    name   = data.get("name","").strip()
    course = data.get("course","").strip()
    if not roll or not name or not course:
        return jsonify({"success":False,"message":"Sab fields required!"}), 400
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

# ─────────────────────────────────────────
# SUBJECT ROUTES
# ─────────────────────────────────────────
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

@app.route("/api/subjects")
def list_subjects():
    conn = get_db()
    try:
        rows = conn.execute("SELECT * FROM subjects ORDER BY subject_name").fetchall()
        return jsonify({"success":True,"subjects":[dict(r) for r in rows]})
    finally:
        conn.close()

# ─────────────────────────────────────────
# ENROLLMENT ROUTES
# ─────────────────────────────────────────
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

# ─────────────────────────────────────────
# FACE CAMERA FUNCTIONS (main thread pe chalti hain)
# ─────────────────────────────────────────
def _do_capture(roll_no, student_name):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return {"success":False,"message":"Camera nahi mila!"}
    count = 0
    try:
        while count < 30:
            ret, frame = cap.read()
            if not ret: continue
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = CASCADE.detectMultiScale(gray, 1.1, 5)
            for (x,y,w,h) in faces:
                face_img = cv2.resize(gray[y:y+h,x:x+w],(200,200))
                face_img = cv2.resize(gray[y:y+h,x:x+w],(200,200))
                face_img = cv2.equalizeHist(face_img)    # yeh add karo
                cv2.imwrite(f"faces/{roll_no}_{count}.jpg", face_img)
                cv2.imwrite(f"faces/{roll_no}_{count}.jpg", face_img)
                count += 1
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,f"Capturing: {count}/30",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            cv2.putText(frame,student_name,(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            cv2.imshow("Face Capture",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if count > 0:
        return {"success":True,"message":f"{count} images captured for {roll_no}!"}
    return {"success":False,"message":"Face detect nahi hua!"}


def _wait_blink(cap, roll_no, timeout=12):
    start = datetime.now()
    eyes_open_count   = 0   # kitne frames mein aankhein khuli thin
    eyes_closed_count = 0   # aankhein band hone ke frames
    blink_confirmed   = False
    phase = "waiting_open"  # phases: waiting_open -> eyes_open -> eyes_closed -> blink_done

    while (datetime.now()-start).seconds < timeout:
        ret, frame = cap.read()
        if not ret: continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(gray, 1.1, 5)

        eye_detected = False
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,140,0),2)
            roi  = gray[y:y+int(h*.55), x:x+w]  # upper 55% of face (eyes area)
            eyes = EYE_CASCADE.detectMultiScale(roi, 1.05, 4, minSize=(15,15), maxSize=(80,80))
            if len(eyes) >= 2:          # DONO aankhein detect honi chahiye (stricter)
                eye_detected = True
                break

        if eye_detected:
            if phase == "waiting_open":
                eyes_open_count += 1
                if eyes_open_count >= 4:   # 4 frames stable open — phase 1 complete
                    phase = "eyes_open"
                    eyes_open_count = 0
            elif phase == "eyes_open":
                eyes_open_count += 1       # keep counting — aankhein khuli hain
                eyes_closed_count = 0
            elif phase == "eyes_closed":
                # aankhein wapas khul gain — BLINK COMPLETE!
                blink_confirmed = True
                break
        else:
            # No eyes detected
            if phase == "eyes_open":
                eyes_closed_count += 1
                if eyes_closed_count >= 2:  # 2 frames band — blink shuru
                    phase = "eyes_closed"
            elif phase == "waiting_open":
                eyes_open_count = 0    # reset agar face nahi mila

        tl = timeout-(datetime.now()-start).seconds
        status_txt = {"waiting_open":"Aankhein dikhao...", "eyes_open":"Ab blink karo!", "eyes_closed":"Aankhein kholna..."}
        cv2.putText(frame, status_txt.get(phase, "BLINK karo!"), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
        cv2.putText(frame, f"{roll_no} | {tl}s", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, "LIVE PERSON VERIFY", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.imshow("Blink Verification",frame)
        cv2.waitKey(1)
    return blink_confirmed


def _scan_face(cap, model, labels, expected_roll, timeout=10):
    start = datetime.now()
    consecutive_matches = 0
    REQUIRED_MATCHES = 3        # 3 baar match hona chahiye
    CONFIDENCE_THRESHOLD = 45   # strict threshold

    while (datetime.now()-start).seconds < timeout:
        ret, frame = cap.read()
        if not ret: continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)   # better recognition
        faces = CASCADE.detectMultiScale(gray, 1.1, 5)
        matched_this_frame = False

        for (x,y,w,h) in faces:
            face_img = cv2.resize(gray[y:y+h,x:x+w],(200,200))
            lbl, conf = model.predict(face_img)
            detected  = labels.get(lbl,"Unknown")
            tl = timeout-(datetime.now()-start).seconds

            if detected == expected_roll and conf < CONFIDENCE_THRESHOLD:
                consecutive_matches += 1
                matched_this_frame = True
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,f"Match {consecutive_matches}/{REQUIRED_MATCHES} (conf:{conf:.0f})",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                if consecutive_matches >= REQUIRED_MATCHES:
                    cv2.imshow("Face Scan",frame)
                    cv2.waitKey(500)
                    return True
            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(frame,f"No match (conf:{conf:.0f})",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        if not matched_this_frame:
            consecutive_matches = 0   # reset if any frame breaks

        tl = timeout-(datetime.now()-start).seconds
        cv2.putText(frame,f"Verifying: {expected_roll} | {tl}s",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,165,255),2)
        cv2.imshow("Face Scan",frame)
        cv2.waitKey(1)
    return False


def _do_attendance(roll_no, subject_id, student, enrolled, already):
    if already:
        return {"success":True,"status":"already_marked","message":f"{roll_no} aaj already mark tha!"}

    model = cv2.face.LBPHFaceRecognizer_create()
    model.read("faces/trainer.yml")
    labels = {}
    with open("faces/labels.txt") as f:
        for line in f:
            line=line.strip()
            if line:
                k,v=line.split(",")
                labels[int(k)]=v

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return {"success":False,"status":"error","message":"Camera nahi mila!"}
    try:
        matched = _scan_face(cap, model, labels, roll_no, timeout=10)
        if not matched:
            return {"success":False,"status":"face_mismatch","message":f"{roll_no} face match nahi hua!"}
        blinked = _wait_blink(cap, roll_no, timeout=10)
        if not blinked:
            return {"success":False,"status":"blink_failed","message":f"{roll_no} blink detect nahi hua!"}
    finally:
        cap.release()
        cv2.destroyAllWindows()

    conn = get_db()
    try:
        today = date.today().isoformat()
        conn.execute("INSERT OR IGNORE INTO attendance(student_id,subject_id,date,time) VALUES(?,?,?,?)",
                     (student["id"],subject_id,today,datetime.now().strftime("%H:%M:%S")))
        conn.commit()
    finally:
        conn.close()

    return {"success":True,"status":"marked","name":student["name"],"roll":roll_no,
            "message":f"{student['name']} ({roll_no}) mark ho gai!"}

# ─────────────────────────────────────────
# FACE ROUTES
# ─────────────────────────────────────────
@app.route("/api/face/capture/<roll_no>")
def capture_face(roll_no):
    conn = get_db()
    try:
        student = conn.execute("SELECT * FROM students WHERE roll_no=?",(roll_no,)).fetchone()
    finally:
        conn.close()
    if not student:
        return jsonify({"success":False,"message":f"{roll_no} nahi mila!"})
    result = run_camera_task(lambda: _do_capture(roll_no, student["name"]))
    return jsonify(result)

@app.route("/api/face/train")
def train_model():
    conn = get_db()
    try:
        students = conn.execute("SELECT * FROM students").fetchall()
    finally:
        conn.close()
    if not students:
        return jsonify({"success":False,"message":"Koi student nahi!"})
    faces, labels, label_map = [], [], {}
    for idx, s in enumerate(students):
        roll = s["roll_no"]
        label_map[idx] = roll
        for f in os.listdir("faces"):
            if f.startswith(roll+"_") and f.endswith(".jpg"):
                img = cv2.imread(os.path.join("faces",f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(cv2.resize(img,(200,200)))
                    labels.append(idx)
    if not faces:
        return jsonify({"success":False,"message":"Pehle faces capture karo!"})
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces, np.array(labels))
    model.save("faces/trainer.yml")
    with open("faces/labels.txt","w") as f:
        for k,v in label_map.items():
            f.write(f"{k},{v}\n")
    return jsonify({"success":True,"message":f"Model trained! {len(faces)} images, {len(students)} students."})

# ─────────────────────────────────────────
# ATTENDANCE
# ─────────────────────────────────────────
@app.route("/api/attendance/single", methods=["POST"])
def attendance_single():
    data       = request.json or {}
    roll_no    = data.get("roll_no","").strip().upper()
    subject_id = data.get("subject_id")
    if not roll_no or not subject_id:
        return jsonify({"success":False,"message":"roll_no aur subject_id required!"}), 400
    if not os.path.exists("faces/trainer.yml"):
        return jsonify({"success":False,"status":"error","message":"Pehle model train karo!"})

    conn = get_db()
    try:
        student  = conn.execute("SELECT * FROM students WHERE roll_no=?",(roll_no,)).fetchone()
        if not student:
            return jsonify({"success":False,"status":"not_found","message":f"{roll_no} nahi mila!"})
        enrolled = conn.execute("""
            SELECT 1 FROM enrollments e JOIN students s ON s.id=e.student_id
            WHERE s.roll_no=? AND e.subject_id=?
        """,(roll_no,subject_id)).fetchone()
        if not enrolled:
            return jsonify({"success":False,"status":"not_enrolled","message":f"{roll_no} enrolled nahi!"})
        already = conn.execute("""
            SELECT 1 FROM attendance WHERE student_id=? AND subject_id=? AND date=?
        """,(student["id"],subject_id,date.today().isoformat())).fetchone()
        student_dict = dict(student)
    finally:
        conn.close()

    result = run_camera_task(lambda: _do_attendance(roll_no, subject_id, student_dict, enrolled, already))
    return jsonify(result)

# ─────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────
@app.route("/api/attendance/report")
def attendance_report():
    subject_id = request.args.get("subject_id", type=int)
    conn = get_db()
    try:
        if subject_id:
            records = conn.execute("""
                SELECT s.roll_no,s.name,s.course,sub.subject_name,a.date,a.time
                FROM attendance a JOIN students s ON a.student_id=s.id
                JOIN subjects sub ON a.subject_id=sub.id WHERE a.subject_id=?
                ORDER BY a.date DESC,a.time DESC
            """,(subject_id,)).fetchall()
        else:
            records = conn.execute("""
                SELECT s.roll_no,s.name,s.course,sub.subject_name,a.date,a.time
                FROM attendance a JOIN students s ON a.student_id=s.id
                JOIN subjects sub ON a.subject_id=sub.id
                ORDER BY a.date DESC,a.time DESC
            """).fetchall()
    finally:
        conn.close()
    data = [dict(r) for r in records]
    csv_file = f"attendance_report_{date.today().isoformat()}.csv"
    with open(csv_file,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["Roll No","Name","Course","Subject","Date","Time"])
        for r in data:
            w.writerow([r["roll_no"],r["name"],r["course"],r["subject_name"],r["date"],r["time"]])
    return jsonify({"success":True,"total":len(data),"csv_saved":csv_file,"records":data})

# ─────────────────────────────────────────
# RUN — Flask thread + main thread camera loop
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  AI Attendance System")
    print("  Browser mein kholo: http://127.0.0.1:5000")
    print("="*50)

    # Flask ko alag thread mein chalao
    flask_thread = threading.Thread(
        target=lambda: app.run(debug=False, use_reloader=False,
                               host="127.0.0.1", port=5000, threaded=True),
        daemon=True
    )
    flask_thread.start()

    # Camera tasks main thread pe chalao (OpenCV ko main thread chahiye)
    print("  Camera loop ready (main thread)")
    process_camera_queue()
