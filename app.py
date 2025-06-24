from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import toml
import logging
import random
import os, re, time, json, requests, nltk, torch, numpy as np, uuid
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from flask_mail import Mail, Message
from email_validator import validate_email, EmailNotValidError
from email.utils import parseaddr
import secrets
import time
import calendar
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from sqlalchemy import func, cast, Date
from os import environ
from dotenv import load_dotenv
load_dotenv()
# ╭─[ A. KONSTANTA ‒ API KEY ]─────────────────────────────────────╮
API_KEY  = os.getenv("FACTCHECK_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
print("[INIT] FACTCHECK_KEY:", bool(API_KEY),
      "| HF_TOKEN:", bool(HF_TOKEN))

HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}",
              "Content-Type": "application/json"}

# --- 1. KONFIGURASI APLIKASI & DATABASE (VERIFIKASI KONFIGURASI) ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'ganti-dengan-kunci-rahasia-anda-yang-panjang-dan-acak')
CORS(app)


history_db = {}

# --- 3. KONFIGURASI FLASK-MAIL & API KEYS ---
# Sangat disarankan untuk memindahkan ini ke environment variables
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('EMAIL_USER', 'pradiptadeskap@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('EMAIL_PASS', 'iyij spyh hvln hkka') # Gunakan App Password 16 digit
app.config['MAIL_DEFAULT_SENDER'] = ('Verify.ai', app.config['MAIL_USERNAME'])


# Inisialisasi Flask-Mail
mail = Mail(app)

# ╭─[ C. NLTK (DIPERBAIKI) ]───────────────────────────────────────╮
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("[NLTK] Paket 'stopwords' tidak ditemukan. Mengunduh...")
    nltk.download("stopwords")
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("[NLTK] Paket 'wordnet' tidak ditemukan. Mengunduh...")
    nltk.download("wordnet")
    
stop_words = set(stopwords.words("indonesian"))
lemmatizer = WordNetLemmatizer()

# ╭─[ D. WEB SCRAPING HELPER ]─────────────────────────────────────╮
def is_url(text: str) -> bool:
    """Mendeteksi apakah teks adalah sebuah URL."""
    return text.strip().startswith(('http://', 'https://'))

def scrape_text_from_url(url: str) -> str | None:
    """Mengambil konten teks utama dari URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=15, headers=headers, allow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        paragraphs = soup.find_all('p')
        if not paragraphs:
             return ' '.join(t.strip() for t in soup.stripped_strings)

        article_text = ' '.join(p.get_text(strip=True) for p in paragraphs)
        
        print(f"[SCRAPE] Berhasil mengambil {len(article_text)} karakter dari {url}")
        return article_text

    except requests.RequestException as e:
        print(f"[SCRAPE] Gagal mengambil URL: {e}")
        return None
    except Exception as e:
        print(f"[SCRAPE] Gagal mem-parsing URL: {e}")
        return None


# ╭─[ E. PRE-PROCESS ]────────────────────────────────────────────╮
EXTRANEOUS_REGEX = re.compile(
    "|".join([
        r"=+\s*\[?\s*(?:kategori|category|penjelasan|referensi)[^\n]*",
        r"selengkapnya\s+di\s+bagian\s+penjelasan\s+dan\s+referensi[^\n]*",
        r"selengkapnya\s+di\s+bagian\s+penjelasan[^\n]*",
        r"selengkapnya\s+dibagian\s+penjelasan\s+dan\s+referensi[^\n]*",
        r"baca\s+selengkapnya\s+di\s+bagian\s+penjelasan[^\n]*",
        r"\(ditulis\s+oleh\s+tim\s+pemeriksa\s+fakta[^\)]*\)",
        r"sumber\s*:.*", r"source\s*:.*"
    ]),
    flags=re.IGNORECASE
)

def clean_text(t: str) -> str:
    t = EXTRANEOUS_REGEX.sub(" ", t.lower())
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"(kompas\.com|turnbackhoax\.id)", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\d+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def preprocess_text(t: str) -> str:
    return " ".join(lemmatizer.lemmatize(w)
                    for w in t.split() if w not in stop_words)

# ╭─[ E-2. WORD ERROR RATE (FITUR BARU) ]──────────────────────────╮
def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Menghitung Word Error Rate (WER) antara teks referensi dan hipotesis.
    WER = (Substitutions + Deletions + Insertions) / Number of Words in Reference
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)
    
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
        
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i, j] = min(d[i-1, j] + 1,          # Deletion
                        d[i, j-1] + 1,          # Insertion
                        d[i-1, j-1] + cost) # Substitution
                        
    n_words = len(ref_words)
    if n_words == 0:
        return float(len(hyp_words) > 0)
    wer = d[len(ref_words), len(hyp_words)] / n_words
    print(f"[WER] Dihitung: {wer:.4f}")
    return wer


# ╭─[ F. MODEL ]───────────────────────────────────────────────────╮
MODEL_DIR = "idbert"
# --- PERUBAHAN DI SINI ---
print(f"[MODEL] Mengunduh atau memuat model '{MODEL_DIR}' dari cache...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=False)
model     = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, local_files_only=False)
# --- AKHIR PERUBAHAN ---
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device); model.eval()
print("[MODEL] Model berhasil dimuat.")
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device); model.eval()

@torch.inference_mode()
def predict_one(text: str):
    t = preprocess_text(clean_text(text))
    inp = tokenizer(t, truncation=True, padding="max_length",
                    max_length=128, return_tensors="pt").to(device)
    logits = model(**inp).logits
    probs  = torch.softmax(logits, -1).cpu().numpy()[0]
    return ("HOAX" if probs.argmax() else "FAKTA", float(probs[1]))

# ╭─[ G. FACT-CHECK helper ]──────────────────────────────────────╮
def _extract_query(txt: str) -> str:
    sent = re.split(r"[.!?\n]", txt, 1)[0][:250]
    return f"\"{sent}\"" if len(sent.split()) > 6 else sent

def check_fact_claims(txt: str):
    if not API_KEY: return []
    query = _extract_query(txt)
    url   = ("https://factchecktools.googleapis.com/v1alpha1/claims:search"
           f"?query={requests.utils.quote(query)}"
           "&languageCode=id,en&pageSize=6"
           f"&key={API_KEY}")
    try:
        r = requests.get(url, timeout=12)
        print("[FACT] status", r.status_code, "| query:", query)
        r.raise_for_status()
        data = r.json()
        claims = []
        for c in data.get("claims", []):
            rev = c.get("claimReview", [])
            if not rev: continue
            first = rev[0]
            claims.append({
                "text"      : c.get("text", "")[:200],
                "publisher" : first.get("publisher", {}).get("name", ""),
                "rating"    : first.get("textualRating", ""),
                "url"       : first.get("url", "")
            })
        print("[FACT] ditemukan:", len(claims))
        return claims
    except Exception as e:
        print("FactCheck error:", e)
        return []

# ╭─[ H. LLM helper (IMPROVED) ]──────────────────────────────────╮
def explain_simple(label, facts):
    base = (f"Berdasarkan analisis AI, teks ini terindikasi "
            f"{'sebagai HOAX' if label=='HOAX' else 'sebagai FAKTA'}.")
    if facts:
        base += f" Ditemukan {len(facts)} klaim terkait di Google Fact Check."
    else:
        base += " Tidak ada klaim serupa yang ditemukan di Google Fact Check."
    return base + " Selalu verifikasi informasi dari sumber terpercaya."

def explain_hf(label, facts, text):
    if not HF_TOKEN:
        return explain_simple(label, facts)

    context = ("Berikut adalah beberapa klaim terkait yang ditemukan dari Google Fact-Check:\n" +
               "\n".join(f"- \"{f['text']}\" (Rating: {f['rating']} oleh {f['publisher']})" for f in facts[:3])
               if facts else "Tidak ada referensi fact-check eksternal yang ditemukan.")

    prompt = (
        "Anda adalah asisten AI analitis yang bertugas memberikan penjelasan mengenai hasil deteksi hoaks.\n\n"
        "== KONTEKS YANG DIBERIKAN ==\n"
        f"1. Teks Pengguna: \"{text[:300]}...\"\n"
        f"2. Hasil Model AI (IndoBERT): Teks ini diberi label sebagai **{label}**.\n"
        f"3. Referensi Google Fact-Check: {context}\n\n"
        "== TUGAS ANDA ==\n"
        "HANYA berdasarkan konteks di atas, berikan penjelasan dalam Bahasa Indonesia yang jelas dalam 2-4 kalimat. Jelaskan mengapa teks tersebut kemungkinan dilabeli demikian, kaitkan dengan referensi jika ada. Akhiri dengan saran singkat yang netral untuk pengguna.\n\n"
        "Penjelasan Anda:"
    )

    models = ["HuggingFaceH4/zephyr-7b-beta", "google/flan-t5-large"]

    for m in models:
        try:
            url = f"https://api-inference.huggingface.co/models/{m}"
            res = requests.post(
                url, headers=HF_HEADERS,
                json={"inputs": prompt,
                      "parameters": {"max_new_tokens": 512, 
                                      "temperature": 0.7,
                                      "return_full_text": False }},
                timeout=30)
            print("[HF ]", m, res.status_code)
            if res.status_code == 503: time.sleep(2); continue
            if res.status_code != 200: continue
            
            js = res.json()
            if isinstance(js, list) and js and "generated_text" in js[0]:
                return js[0]["generated_text"].strip()
            if isinstance(js, dict) and "generated_text" in js:
                 return js["generated_text"].strip()

        except Exception as e:
            print("HF err:", e)
            continue

    return explain_simple(label, facts)


# Muat konfigurasi dari file toml
try:
    config = toml.load('config.toml')
    db_config = config['database']
    db_uri = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
    logging.debug(f"Database URI: {db_uri}")  # Debug log to check connection
except (FileNotFoundError, KeyError) as e:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fallback.db'
    logging.error(f"Error memuat konfigurasi database dari config.toml: {e}")


# Menonaktifkan pelacakan perubahan objek yang tidak perlu
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inisialisasi Database
db = SQLAlchemy(app)





# --- Menambahkan Kelas User ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    otp_hash = db.Column(db.String(255), nullable=True)
    otp_expiry = db.Column(db.DateTime, nullable=True)
    avatar_url = db.Column(db.String(255), nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)




    def __repr__(self):
        return f'<User  {self.username}>'
# Tambahkan setelah kelas User
class Prediction(db.Model):
    # ID unik untuk setiap prediksi, menggunakan UUID
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Kunci asing untuk menghubungkan prediksi ini ke seorang pengguna
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Teks asli yang dimasukkan oleh pengguna (bisa kalimat atau URL)
    input_text = db.Column(db.Text, nullable=False)
    
    # Teks yang ditampilkan (hasil scraping jika input adalah URL)
    displayed_text = db.Column(db.Text, nullable=False)

    # Hasil dari model
    label = db.Column(db.String(10), nullable=False)
    prob_hoax = db.Column(db.Float, nullable=False)
    explanation = db.Column(db.Text)
    
    # Simpan hasil fact-check sebagai string JSON untuk fleksibilitas
    fact_checks_json = db.Column(db.Text)
    
    # Timestamp kapan prediksi dibuat
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    processing_duration_ms = db.Column(db.Integer, nullable=True)
    wer_score = db.Column(db.Float, nullable=True)

    # Relasi agar kita bisa memanggil user dari objek prediksi
    user = db.relationship('User', backref=db.backref('predictions', lazy=True, order_by="Prediction.created_at.desc()"))

    def to_dict(self):
        """Mengubah objek menjadi dictionary untuk respons JSON."""
        return {
            "id": self.id,
            "text": self.displayed_text,
            "label": self.label,
            "prob_hoax": self.prob_hoax,
            "explanation": self.explanation,
            "fact_checks": json.loads(self.fact_checks_json or '[]'),
            "processing_duration_ms": self.processing_duration_ms,
            "wer_score": self.wer_score
        }
# --- Setup Logging ---
logging.basicConfig(level=logging.DEBUG)

# Daftar gaya rambut dari dokumentasi Avataaars
MALE_TOPS = [
    'ShortHairShortFlat', 'ShortHairShortRound', 'ShortHairTheCaesar', 'ShortHairFrizzle',
    'ShortHairSides', 'ShortHairDreads01', 'ShortHairDreads02'
]
FEMALE_TOPS = [
    'LongHairStraight', 'LongHairStraight2', 'LongHairBob', 'LongHairMiaWallace',
    'LongHairBigHair', 'LongHairCurvy', 'LongHairFrida', 'LongHairNotTooLong'
]
# DAFTAR BARU UNTUK FITUR WAJAH
MALE_FACIAL_HAIR = [
    'Blank', 'BeardLight', 'BeardMedium', 'MoustacheFancy', 'Blank', 'BeardMajestic', 'Blank'
] # 'Blank' diulang agar tidak semua laki-laki punya jenggot/kumis

MALE_EYEBROWS = ['Default', 'DefaultNatural', 'RaisedExcited', 'UpDown']
FEMALE_EYEBROWS = ['DefaultNatural', 'UpDown', 'RaisedExcitedHappy']


def generate_avatar_url(user_obj):
    """
    Membuat URL avatar yang konsisten dan lebih detail sesuai gender untuk seorang pengguna.
    """
    seed = user_obj.username
    base_url = "https://avataaars.io/?avatarStyle=Transparent&skinColor=Light"
    params = f"&seed={seed}"

    if user_obj.gender == 'Laki-laki':
        top_style = MALE_TOPS[len(seed) % len(MALE_TOPS)]
        facial_hair_style = MALE_FACIAL_HAIR[user_obj.id % len(MALE_FACIAL_HAIR)]
        eyebrow_style = MALE_EYEBROWS[user_obj.id % len(MALE_EYEBROWS)]
        
        params += f"&topType={top_style}"
        params += f"&facialHairType={facial_hair_style}"
        params += f"&eyebrowType={eyebrow_style}"
        params += "&clotheType=ShirtVNeck"

    elif user_obj.gender == 'Perempuan':
        top_style = FEMALE_TOPS[len(seed) % len(FEMALE_TOPS)]
        eyebrow_style = FEMALE_EYEBROWS[user_obj.id % len(FEMALE_EYEBROWS)]
        
        params += f"&topType={top_style}"
        params += "&facialHairType=Blank"  # Perempuan tidak memiliki rambut wajah
        params += f"&eyebrowType={eyebrow_style}"
        params += "&clotheType=BlazerShirt"
    
    else:
        # Fallback jika gender tidak diatur
        params += "&topType=ShortHairShortFlat"

    return base_url + params

# Route untuk halaman utama (index.html)
@app.route('/')
def home():
    # Logika sederhana: jika pengguna sudah login, arahkan ke dashboard.
    # Jika belum, arahkan ke halaman login.
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

# GANTI FUNGSI LAMA ANDA DENGAN YANG INI DI app.py

@app.route('/dashboard')
def dashboard():
    # 1. Cek sesi pengguna
    if 'user_id' not in session:
        flash('Anda harus login terlebih dahulu untuk mengakses halaman ini.', 'warning')
        return redirect(url_for('login'))
    
    # 2. Ambil objek user lengkap
    user = User.query.get(session['user_id'])
    
    # 3. Pengaman sesi
    if not user:
        session.clear()
        flash('Sesi tidak valid, silakan login kembali.', 'danger')
        return redirect(url_for('login'))

    # 4. Ambil riwayat prediksi untuk pengguna ini dari database
    #    Relasi 'predictions' yang kita buat di model akan otomatis mengambil data ini.
    #    Data sudah diurutkan dari yang terbaru berkat 'order_by' di backref.
    user_history = user.predictions 

    # 5. Kirim objek 'user' dan 'history' ke template
    return render_template('dashboard.html', user=user, history=user_history)

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Jika pengguna sudah login (sudah ada session), langsung arahkan ke dashboard
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    # Jika metode request adalah POST (pengguna menekan tombol "Masuk")
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Validasi sederhana agar tidak kosong
        if not email or not password:
            flash('Email dan password harus diisi.', 'danger')
            return redirect(url_for('login'))

        # Cari pengguna di database berdasarkan email
        user = User.query.filter_by(email=email).first()

        # Cek apakah pengguna ada DAN passwordnya cocok
        if user and check_password_hash(user.password, password):
            # ---- BAGIAN KUNCI ----
            # Jika berhasil, simpan ID dan username pengguna ke dalam session
            session['user_id'] = user.id
            session['username'] = user.username
            user.last_seen = datetime.utcnow()
            db.session.commit()
            flash(f'Selamat datang kembali, {user.username}!', 'success')
            # Arahkan (redirect) ke fungsi/route 'dashboard'
            return redirect(url_for('dashboard'))
        else:
            # Jika pengguna tidak ada atau password salah
            flash('Email atau password salah. Silakan coba lagi.', 'danger')
            return redirect(url_for('login'))

    # Jika metode adalah GET, tampilkan halaman login biasa
    return render_template('login.html')



@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # ------------------------- ambil data form -------------------------
        username = request.form.get('username')
        email    = request.form.get('email')
        password = request.form.get('password')
        gender   = request.form.get('gender')

        logging.debug(f"Received signup data - Username: {username}, "
                      f"Email: {email}, Gender: {gender}")

        # ------------------------- validasi dasar --------------------------
        if not username or not email or not password:
            flash('Semua field wajib diisi.', 'warning')
            return redirect(url_for('signup'))

        if not gender:
            flash('Anda harus memilih gender.', 'warning')
            return redirect(url_for('signup'))

        if '@' not in email or '.' not in email:
            flash('Email tidak valid. Pastikan format email benar.', 'warning')
            return redirect(url_for('signup'))

        if User.query.filter_by(email=email).first():
            flash('Maaf, email sudah terdaftar.', 'warning')
            return redirect(url_for('signup'))

        if User.query.filter_by(username=username).first():
            flash('Username sudah digunakan.', 'warning')
            return redirect(url_for('signup'))

        if len(password) < 8:
            flash('Password harus lebih dari 8 karakter.', 'warning')
            return redirect(url_for('signup'))

        if (not any(c.isupper() for c in password) or
            not any(c.isdigit() for c in password) or
            not any(c in '!@#$%^&*(),.?":{}|<>' for c in password)):
            flash('Password harus mengandung huruf kapital, angka, dan simbol.',
                  'warning')
            return redirect(url_for('signup'))

        # ----------------------- buat & simpan user ------------------------
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        new_user = User(
            username=username,
            email=email,
            password=hashed_password,
            gender=gender
        )

        db.session.add(new_user)

        try:
            db.session.commit()  # commit pertama agar new_user.id tersedia

            # generate avatar setelah id tersedia
            if not new_user.avatar_url:
                new_user.avatar_url = generate_avatar_url(new_user)
                db.session.commit()  # commit kedua untuk simpan avatar_url

            logging.debug("User berhasil disimpan ke database.")
            flash('Pendaftaran berhasil! Silakan login.', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            db.session.rollback()
            logging.error(f"Gagal menyimpan data: {e}")
            flash(f"Gagal menyimpan data: {e}", 'danger')
            return redirect(url_for('signup'))

    # ------------------------- GET request -------------------------
    return render_template('signup.html', user=None)




# Di dalam app.py

@app.route('/lupapasword', methods=['GET', 'POST'])
def lupapasword():
    if request.method == 'POST':
        email = request.form.get('email')

        if not email or '@' not in email:
            flash("Format email tidak valid.", 'danger')
            return redirect(url_for('lupapasword'))

        user = User.query.filter_by(email=email).first()

        if user:
            try:
                otp = "".join(secrets.choice("0123456789") for _ in range(6))
                
                # --- TAMBAHKAN PRINT DI SINI ---
                print(f"--- DEBUG LUPAPASWORD ---")
                print(f"User: {user.email}")
                print(f"OTP yang Dibuat: {otp}")
                # -----------------------------
                
                user.otp_hash = generate_password_hash(otp)
                user.otp_expiry = datetime.utcnow() + timedelta(minutes=10)
                
                # --- TAMBAHKAN PRINT LAGI DI SINI ---
                print(f"Hash yang AKAN DISIMPAN: {user.otp_hash}")
                # -----------------------------------
                
                db.session.commit() # Coba simpan ke DB

                # --- TAMBAHKAN PRINT SETELAH COMMIT ---
                print(f"Commit ke database selesai.")
                # -----------------------------------

                session['email_for_reset'] = user.email

                msg = Message(
                    subject="Kode Verifikasi untuk Reset Password Verify.ai",
                    recipients=[user.email]
                )
                msg.html = f"<h3>Gunakan kode berikut untuk mereset password Anda: <b>{otp}</b></h3><p>Kode ini hanya berlaku selama 10 menit.</p>"
                mail.send(msg)
                
                print(f"Email berhasil dikirim ke {user.email}")
                print(f"---------------------------\n")


            except Exception as e:
                logging.error(f"GAGAL DALAM PROSES LUPA PASSWORD untuk {email}: {str(e)}")
        
        flash('Jika email Anda terdaftar, sebuah kodeverifikasi telah dikirim.', 'success')
        return redirect(url_for('otplupapassword'))

    return render_template('lupapasword.html')

@app.route('/otplupapassword', methods=['GET', 'POST'])
def otplupapassword():
    if 'email_for_reset' not in session:
        return redirect(url_for('lupapasword'))

    if request.method == 'POST':
        submitted_otp = request.form.get('otp')
        email = session['email_for_reset']
        
        user = User.query.filter_by(email=email).first()

        # --- TAMBAHKAN PRINT DI SINI ---
        print(f"\n--- DEBUG OTPLUPAPASSWORD ---")
        print(f"User: {email}")
        print(f"OTP yang Dimasukkan: {submitted_otp}")
        print(f"Hash yang DIBACA dari DB: {user.otp_hash}")
        print(f"-----------------------------\n")
        # -----------------------------

        if not user or not user.otp_hash or not user.otp_expiry:
            flash('Terjadi kesalahan. Silakan ulangi proses lupa password.', 'danger')
            return redirect(url_for('lupapasword'))

        if datetime.utcnow() > user.otp_expiry:
            flash('Kode OTP sudah kedaluwarsa. Silakan minta yang baru.', 'warning')
            return redirect(url_for('lupapasword'))
            
        is_valid = check_password_hash(user.otp_hash, submitted_otp)

        # --- TAMBAHKAN PRINT HASIL VERIFIKASI ---
        print(f"Hasil check_password_hash: {is_valid}")
        # --------------------------------------

        if is_valid:
            # ... (kode sukses) ...
            session['user_verified_for_reset'] = True
            user.otp_hash = None
            user.otp_expiry = None
            db.session.commit()
            flash('Verifikasi berhasil! Silakan masukkan password baru Anda.', 'success')
            return redirect(url_for('reset_with_otp'))
        else:
            flash('Kode OTP salah. Silakan coba lagi.', 'danger')
            return redirect(url_for('otplupapassword'))

    return render_template('otplupapassword.html')

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_with_otp():
    # Pastikan user sudah melewati verifikasi OTP
    if not session.get('user_verified_for_reset'):
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash('Password dan konfirmasi password tidak cocok.', 'warning')
            return redirect(url_for('reset_with_otp'))
        
        # Validasi kekuatan password (copy dari halaman signup)
        if len(password) < 8:
            flash('Password harus lebih dari 8 karakter.', 'warning')
            return redirect(url_for('reset_with_otp'))

        email = session['email_for_reset']
        user = User.query.filter_by(email=email).first_or_404()
        
        # Update password user
        user.password = generate_password_hash(password, method='pbkdf2:sha256')
        db.session.commit()
        
        # Hapus session agar bersih
        session.pop('email_for_reset', None)
        session.pop('user_verified_for_reset', None)
        
        flash('Password Anda telah berhasil direset! Silakan login dengan password baru.', 'success')
        return redirect(url_for('login'))

    return render_template('reset_password.html') # Buat file HTML baru untuk ini

# TAMBAHKAN BLOK KODE BARU INI
@app.route('/logout')
def logout():
    # Hapus semua data dari session
    session.clear()
    flash('Anda telah berhasil logout.', 'info')
    return redirect(url_for('login'))

@app.route('/profil', methods=['GET', 'POST'])
def profil():
    if 'user_id' not in session:
        flash('Anda harus login untuk melihat halaman profil.', 'warning')
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    if not user:
        flash('Pengguna tidak ditemukan.', 'danger')
        session.clear()
        return redirect(url_for('login'))

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'update_profil':
            new_username = request.form.get('username')
            new_email = request.form.get('email')

            existing_user = User.query.filter(User.username == new_username, User.id != user.id).first()
            if existing_user:
                flash('Username tersebut sudah digunakan. Silakan pilih yang lain.', 'warning')
                return redirect(url_for('profil'))

            user.username = new_username
            user.email = new_email
            db.session.commit()
            flash('Profil berhasil diperbarui!', 'success')

        elif action == 'change_password':
            current_password = request.form.get('current_password')
            new_password = request.form.get('new_password')
            confirm_password = request.form.get('confirm_password')

            if not check_password_hash(user.password, current_password):
                flash('Password Anda saat ini salah.', 'danger')
                return redirect(url_for('profil'))
            
            if not new_password or len(new_password) < 8:
                flash('Password baru harus minimal 8 karakter.', 'warning')
                return redirect(url_for('profil'))
            
            if new_password != confirm_password:
                flash('Konfirmasi password baru tidak cocok.', 'warning')
                return redirect(url_for('profil'))

            user.password = generate_password_hash(new_password, method='pbkdf2:sha256')
            db.session.commit()
            flash('Password berhasil diubah!', 'success')

        return redirect(url_for('profil'))

    return render_template('profil.html', user=user)

# ROUTE BARU UNTUK DATA ANALISIS
@app.route("/analysis", methods=["GET"])
def analysis():
    hoax_count = 0
    fakta_count = 0
    for item in history_db.values():
        if item.get("label") == "HOAX":
            hoax_count += 1
        elif item.get("label") == "FAKTA":
            fakta_count += 1
    
    return jsonify({
        "hoax": hoax_count,
        "fakta": fakta_count
    })

@app.route("/predict", methods=["POST"])
def predict():
    # Pastikan pengguna sudah login untuk menyimpan riwayat
    if 'user_id' not in session:
        return jsonify({"error": "Akses ditolak. Silakan login terlebih dahulu."}), 401
    # --- MULAI MENGHITUNG WAKTU ---
    start_time = time.perf_counter()
    try:
        user = User.query.get(session['user_id'])
        if user:
            user.last_seen = datetime.utcnow()
            db.session.commit()
    except Exception as e:
        logging.error(f"Gagal update last_seen untuk user {session['user_id']}: {e}")
        db.session.rollback()

    data = request.get_json(silent=True) or {}
    user_input = (data.get("text") or "").strip()
    if not user_input:
        return jsonify({"error": "Input teks atau URL kosong"}), 400

    text_to_analyze = user_input
    wer_score = None # Inisialisasi WER

    if is_url(user_input):
        print(f"[PREDICT] Terdeteksi URL: {user_input}")
        scraped_content = scrape_text_from_url(user_input)
        if scraped_content:
            text_to_analyze = scraped_content
            # Hitung WER antara teks asli hasil scrape dengan versi yang sudah diproses
            wer_hypothesis = preprocess_text(clean_text(scraped_content))
            wer_score = calculate_wer(scraped_content.lower(), wer_hypothesis)
        else:
            return jsonify({"error": f"Gagal mengambil atau memproses konten dari URL: {user_input}"}), 400
    
    # Menjalankan pipeline ML (tidak ada perubahan di sini)
    label, prob = predict_one(text_to_analyze)
    facts       = check_fact_claims(text_to_analyze)
    explain     = explain_hf(label, facts, text_to_analyze)
    
    # --- SELESAI MENGHITUNG WAKTU ---
    end_time = time.perf_counter()
    # Hitung durasi dalam milidetik (ms)
    duration_ms = (end_time - start_time) * 1000

    # --- BAGIAN KRUSIAL: SIMPAN KE DATABASE ---
    try:
        new_prediction = Prediction(
            user_id=session['user_id'],
            input_text=user_input,
            displayed_text=text_to_analyze,
            label=label,
            prob_hoax=prob,
            explanation=explain,
            # Simpan 'facts' sebagai string JSON
            fact_checks_json=json.dumps(facts),
            processing_duration_ms=int(duration_ms),
            wer_score=wer_score # Simpan WER ke DB
        )
        db.session.add(new_prediction)
        db.session.commit()
        
        # Kirim kembali data yang baru dibuat ke frontend
        # Kita menggunakan to_dict() agar formatnya sama dengan yang diharapkan JS
        return jsonify(new_prediction.to_dict())

    except Exception as e:
        db.session.rollback()
        logging.error(f"Gagal menyimpan prediksi ke DB: {e}")
        return jsonify({"error": "Terjadi kesalahan saat menyimpan hasil."}), 500

@app.route("/history/<history_id>", methods=["GET"])
def get_history_item(history_id):
    if 'user_id' not in session:
        return jsonify({"error": "Akses ditolak."}), 401

    # Ambil item dari database
    item = Prediction.query.get(history_id)
    
    # Jika tidak ditemukan ATAU item ini bukan milik user yang login
    if not item or item.user_id != session['user_id']:
        return jsonify({"error": "Riwayat tidak ditemukan atau Anda tidak memiliki akses."}), 404
    
    # Kembalikan data dalam format JSON yang benar
    return jsonify(item.to_dict())

@app.route("/api/userstats")
def get_user_stats():
    users = User.query.order_by(User.id.desc()).limit(5).all()  # Ambil 5 pengguna terakhir
    avatars = [u.avatar_url or '/static/default-avatar.png' for u in users]

    return jsonify({
        "total_users": User.query.count(),
        "avatars": avatars
    })
@app.route("/api/user-growth-stats")
def get_user_growth_stats():
    try:
        # --- Bagian 1: Menghitung Total Pengguna ---
        total_users = db.session.query(func.count(User.id)).scalar()

        # --- Bagian 2: Menghitung Pertumbuhan Pengguna Baru (30 hari vs 30 hari sebelumnya) ---
        today = datetime.utcnow()
        start_current_month = today - timedelta(days=30)
        start_previous_month = today - timedelta(days=60)

        new_users_current_month = db.session.query(func.count(User.id)).filter(
            User.created_at.between(start_current_month, today)
        ).scalar()

        new_users_previous_month = db.session.query(func.count(User.id)).filter(
            User.created_at.between(start_previous_month, start_current_month)
        ).scalar()
        
        percentage_change = 0
        if new_users_previous_month > 0:
            change = new_users_current_month - new_users_previous_month
            percentage_change = (change / new_users_previous_month) * 100
        elif new_users_current_month > 0:
            percentage_change = 100  # Pertumbuhan tak terhingga, anggap 100%

        # --- Bagian 3: Menyiapkan Data untuk Grafik Garis (30 hari terakhir) ---
        # Query efisien untuk menghitung pendaftaran per hari
        daily_signups_query = db.session.query(
            cast(User.created_at, Date).label('date'),
            func.count(User.id).label('count')
        ).filter(
            User.created_at >= start_current_month
        ).group_by('date').order_by('date').all()
        
        # Buat dictionary untuk mapping tanggal -> jumlah
        signups_map = {str(date): count for date, count in daily_signups_query}

        chart_labels = []
        chart_data = []
        # Loop 30 hari ke belakang untuk memastikan semua hari ada (termasuk yang 0 pendaftar)
        for i in range(30):
            day = start_current_month.date() + timedelta(days=i)
            chart_labels.append(day.strftime('%d/%m'))
            chart_data.append(signups_map.get(str(day), 0))
            
        # --- Bagian 4: Mengemas Semua Data untuk Dikirim ---
        return jsonify({
            "total_users": total_users,
            "new_users_current_month": new_users_current_month,
            "percentage_change": percentage_change,
            "chart_labels": chart_labels,
            "chart_data": chart_data
        })

    except Exception as e:
        logging.error(f"Error pada get_user_growth_stats: {e}")
        return jsonify({"error": "Gagal mengambil data statistik pertumbuhan"}), 500
# Di dalam app.py

@app.route("/api/active-user-gender-stats")
def get_active_user_gender_stats():
    """
    API untuk menyediakan data pengguna aktif berdasarkan gender DAN total pertumbuhannya.
    Definisi aktif: terlihat dalam 2 HARI TERAKHIR.
    """
    try:
        # Tentukan periode waktu
        current_period_start = datetime.utcnow() - timedelta(days=2)
        previous_period_start = datetime.utcnow() - timedelta(days=4)

        # Hitung pengguna aktif berdasarkan gender untuk periode saat ini (2 hari terakhir)
        male_active = db.session.query(func.count(User.id)).filter(
            User.gender == 'Laki-laki',
            User.last_seen >= current_period_start
        ).scalar() or 0

        female_active = db.session.query(func.count(User.id)).filter(
            User.gender == 'Perempuan',
            User.last_seen >= current_period_start
        ).scalar() or 0
        
        total_active_now = male_active + female_active

        # --- BAGIAN PENTING YANG HILANG (KITA TAMBAHKAN KEMBALI) ---
        # Hitung total pengguna aktif di periode SEBELUMNYA (4-2 hari yang lalu)
        total_active_previously = db.session.query(func.count(User.id)).filter(
            User.last_seen.between(previous_period_start, current_period_start)
        ).scalar() or 0

        # Hitung persentase perubahan
        percentage_change = 0.0
        if total_active_previously > 0:
            change = total_active_now - total_active_previously
            percentage_change = (change / total_active_previously) * 100
        elif total_active_now > 0:
            # Jika sebelumnya 0 dan sekarang ada, anggap pertumbuhan 100%
            percentage_change = 100.0
        # -----------------------------------------------------------

        # Kembalikan semua data yang dibutuhkan, TERMASUK percentage_change
        return jsonify({
            "total_active": total_active_now,
            "male_active": male_active,
            "female_active": female_active,
            "percentage_change": percentage_change  # <-- Data penting ini sekarang dikirim
        })

    except Exception as e:
        logging.error(f"Error pada get_active_user_gender_stats: {e}")
        return jsonify({"error": "Gagal mengambil data statistik gender"}), 500

# Di dalam app.py

@app.route("/api/avg-duration-stats")
def get_avg_duration_stats():
    """
    API untuk menghitung rata-rata durasi pemrosesan, membandingkan
    30 hari terakhir dengan 30 hari sebelumnya. (Versi diperbaiki)
    """
    try:
        # Tentukan periode waktu
        today = datetime.utcnow()
        current_period_start = today - timedelta(days=30)
        previous_period_start = today - timedelta(days=60)

        # --- PERBAIKAN: Secara eksplisit ubah hasil query ke float ---
        avg_duration_current_raw = db.session.query(
            func.avg(Prediction.processing_duration_ms)
        ).filter(
            Prediction.created_at >= current_period_start,
            Prediction.processing_duration_ms.isnot(None)
        ).scalar()
        
        # Jika hasilnya None (tidak ada data), jadikan 0. Jika ada, ubah ke float.
        avg_duration_current = float(avg_duration_current_raw or 0.0)

        avg_duration_previous_raw = db.session.query(
            func.avg(Prediction.processing_duration_ms)
        ).filter(
            Prediction.created_at.between(previous_period_start, current_period_start),
            Prediction.processing_duration_ms.isnot(None)
        ).scalar()

        avg_duration_previous = float(avg_duration_previous_raw or 0.0)
        # --- AKHIR PERBAIKAN ---
        
        # Hitung persentase perubahan
        percentage_change = 0.0
        if avg_duration_previous > 0:
            change = avg_duration_current - avg_duration_previous
            percentage_change = (change / avg_duration_previous) * 100
        elif avg_duration_current > 0:
            percentage_change = 0.0

        return jsonify({
            "avg_duration_ms": int(avg_duration_current),
            "percentage_change": percentage_change
        })

    except Exception as e:
        # PENTING: Cetak error ini ke terminal Anda untuk debugging
        logging.error(f"Error pada get_avg_duration_stats: {e}") 
        return jsonify({"error": "Gagal mengambil data durasi"}), 500

@app.route("/api/monthly-prediction-summary")
def get_monthly_prediction_summary():
    """
    API untuk menyediakan ringkasan prediksi (Hoax vs Fakta) per minggu
    untuk bulan berjalan.
    """
    try:
        # Menggunakan tanggal saat ini untuk menentukan bulan berjalan
        today = datetime.utcnow().date()
        # Menentukan hari pertama dan terakhir dari bulan ini
        start_of_month = today.replace(day=1)
        _, num_days_in_month = calendar.monthrange(today.year, today.month)
        end_of_month = today.replace(day=num_days_in_month)

        # 1. Hitung total prediksi selama sebulan ini
        total_predictions_month = db.session.query(func.count(Prediction.id)).filter(
            Prediction.created_at.between(start_of_month, end_of_month + timedelta(days=1))
        ).scalar() or 0

        # 2. Siapkan struktur data untuk 4 minggu
        weekly_summary = {
            "labels": ["Minggu 1", "Minggu 2", "Minggu 3", "Minggu 4"],
            "hoax_counts": [0, 0, 0, 0],
            "fakta_counts": [0, 0, 0, 0]
        }

        # Query semua prediksi di bulan ini untuk diproses
        predictions_this_month = Prediction.query.filter(
            Prediction.created_at.between(start_of_month, end_of_month + timedelta(days=1))
        ).all()

        # Kelompokkan ke dalam 4 minggu secara manual di Python
        # Ini adalah cara yang paling kompatibel untuk semua jenis database
        for pred in predictions_this_month:
            day = pred.created_at.day
            week_index = -1
            if 1 <= day <= 7: week_index = 0    # Minggu 1
            elif 8 <= day <= 14: week_index = 1   # Minggu 2
            elif 15 <= day <= 21: week_index = 2  # Minggu 3
            elif 22 <= day <= 31: week_index = 3  # Minggu 4 (dan sisa hari)

            if week_index != -1:
                if pred.label == "HOAX":
                    weekly_summary["hoax_counts"][week_index] += 1
                elif pred.label == "FAKTA":
                    weekly_summary["fakta_counts"][week_index] += 1
        
        # 3. Kembalikan data yang sudah terstruktur dengan rapi
        return jsonify({
            "total_predictions_month": total_predictions_month,
            "weekly_summary": weekly_summary
        })

    except Exception as e:
        logging.error(f"Error pada get_monthly_prediction_summary: {e}")
        return jsonify({"error": "Gagal mengambil ringkasan data bulanan"}), 500


@app.route("/api/accuracy-by-label")
def get_accuracy_by_label():
    """
    API untuk menghitung rata-rata akurasi (keyakinan model)
    secara terpisah untuk label HOAX dan FAKTA.
    """
    try:
        # 1. Hitung rata-rata akurasi untuk prediksi HOAX
        # Akurasi untuk HOAX adalah nilai prob_hoax itu sendiri.
        avg_accuracy_hoax_raw = db.session.query(func.avg(Prediction.prob_hoax)).filter(
            Prediction.label == 'HOAX'
        ).scalar()
        avg_accuracy_hoax = (float(avg_accuracy_hoax_raw or 0.0)) * 100

        # 2. Hitung rata-rata akurasi untuk prediksi FAKTA
        # Akurasi untuk FAKTA adalah 1.0 - prob_hoax.
        # Kita menggunakan func.avg(1.0 - Prediction.prob_hoax)
        avg_accuracy_fakta_raw = db.session.query(func.avg(1.0 - Prediction.prob_hoax)).filter(
            Prediction.label == 'FAKTA'
        ).scalar()
        avg_accuracy_fakta = (float(avg_accuracy_fakta_raw or 0.0)) * 100
        
        return jsonify({
            "avg_accuracy_hoax": avg_accuracy_hoax,
            "avg_accuracy_fakta": avg_accuracy_fakta
        })

    except Exception as e:
        logging.error(f"Error pada get_accuracy_by_label: {e}")
        return jsonify({"error": "Gagal mengambil data akurasi"}), 500

# Menjalankan aplikasi jika file ini dieksekusi secara langsung
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Membuat tabel jika belum ada
    app.run(debug=True, host='0.0.0.0', port=int(environ.get('PORT', 5000))))




