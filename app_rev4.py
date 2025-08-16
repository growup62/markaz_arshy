import sqlite3
import os
import time
import logging
import threading
import json
import hashlib
import uuid
from datetime import datetime, timedelta, date
from flask import Flask, jsonify, request, render_template, session, redirect, url_for, send_from_directory, flash, g, current_app
from flask_socketio import SocketIO
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import secrets
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField, HiddenField
from wtforms.validators import DataRequired, EqualTo, Regexp
from wtforms import BooleanField
from dateutil.relativedelta import relativedelta
from functools import wraps

app = Flask(__name__)
# --- PERUBAHAN: Memuat secret key dari environment variable untuk keamanan ---
app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(24))
app.config['INTERNAL_SECRET_KEY'] = app.secret_key
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")
DATABASE_FILE = 'users.db'
UPLOAD_FOLDER = 'static/bukti_pembayaran'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# --- Global States ---
INTERNAL_SECRET_KEY = os.environ.get('a8c3e6d2b7f1a0e9d8c5b3a2f1e0d9c7b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e9')
if not INTERNAL_SECRET_KEY:
    logging.critical("FATAL: INTERNAL_SECRET_KEY environment variable not set.")
    # Di lingkungan produksi, Anda mungkin ingin keluar dari aplikasi di sini
    # exit(1)

last_signal_info = {}
signal_context_cache = {}
daily_signal_counts = {} # Key: api_key, Value: {'date': 'YYYY-MM-DD', 'count': X}
feedback_file_lock = threading.Lock()
db_write_lock = threading.Lock()
open_positions_map = {}
open_positions_lock = threading.Lock()
SYMBOL_ALIAS_MAP = {
    'XAUUSD': 'XAUUSD', 'XAUUSDc': 'XAUUSD', 'XAUUSDm': 'XAUUSD', 'GOLD': 'XAUUSD',
    'BTCUSD': 'BTCUSD', 'BTCUSDc': 'BTCUSD', 'BTCUSDm': 'BTCUSD',
}

# --- Helper Functions ---
def get_user_status(api_key: str) -> str:
    """Mendapatkan status user ('active', 'trial', 'expired', 'invalid') dari API key."""
    if not api_key:
        return 'invalid'
    
    db = get_db()
    user = db.execute("SELECT end_date, status FROM users WHERE api_key = ?", (api_key,)).fetchone()
    
    if not user:
        return 'invalid'
    
    if user['status'] not in ['active', 'trial']:
        return user['status']

    license_end = datetime.strptime(user['end_date'], '%Y-%m-%d').date()
    if date.today() > license_end:
        return 'expired'
        
    return user['status']

def get_open_positions(api_key, symbol):
    key = f"{api_key}_{symbol}"
    with open_positions_lock:
        return open_positions_map.get(key, [])

def is_too_close_to_open_position(new_entry, open_positions, pip_threshold=100.0, max_age_hours=8):
    now = datetime.now()
    recent_positions = []
    for pos in open_positions:
        try:
            pos_time = datetime.fromisoformat(pos['time'])
            if (now - pos_time) < timedelta(hours=max_age_hours):
                recent_positions.append(pos)
        except (ValueError, KeyError):
            recent_positions.append(pos)

    for pos in recent_positions:
        try:
            if abs(float(pos['entry']) - float(new_entry)) < pip_threshold:
                return True
        except (ValueError, TypeError):
            continue
    return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_signal_id(api_key, order_type, timestamp):
    data = f"{api_key}:{order_type}:{timestamp}"
    return hashlib.md5(data.encode()).hexdigest()

def get_user_license_details(user_data):
    end_date_obj = datetime.strptime(user_data['end_date'], '%Y-%m-%d').date()
    today = date.today()
    status = user_data['status']

    if status == 'pending_activation':
        return "Menunggu Aktivasi", "bg-warning"
    elif today <= end_date_obj and status in ['active', 'trial']:
        return "Aktif", "bg-success"
    else:
        return "Kadaluarsa", "bg-danger"

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE_FILE, check_same_thread=False, timeout=30)
    g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db_data():
    with app.app_context():
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                api_key TEXT UNIQUE NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'trial',
                proof_filename TEXT DEFAULT NULL,
                duration_pending INTEGER DEFAULT NULL,
                whatsapp_number TEXT UNIQUE
            )
        ''')
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN whatsapp_number TEXT")
        except sqlite3.OperationalError:
            pass
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        cursor.execute("SELECT COUNT(*) FROM admins WHERE username = 'admin'")
        if cursor.fetchone()[0] == 0:
            default_admin_password = generate_password_hash('admin123')
            cursor.execute("INSERT INTO admins (username, password) VALUES (?, ?)", ('admin', default_admin_password))
        conn.commit()

# === FORM CLASSES ===
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Ulangi Password', validators=[DataRequired(), EqualTo('password', message='Password dan Ulangi Password harus sama')])
    whatsapp_number = StringField('No. WhatsApp', validators=[DataRequired(), Regexp(r'^\+\d{8,16}$', message='Format Nomor WhatsApp tidak valid (contoh: +6281234567890)')])
    agree_terms = BooleanField('Saya menyetujui Syarat & Ketentuan', validators=[DataRequired(message='Anda harus menyetujui Syarat & Ketentuan')])
    submit = SubmitField('Register')

class SubscribeForm(FlaskForm):
    duration = HiddenField('Duration', validators=[DataRequired()])
    proof_file = FileField('Unggah Bukti Pembayaran', validators=[DataRequired()])
    submit = SubmitField('Kirim Bukti Pembayaran')

# === ROUTES ===
@app.before_request
def require_login():
    # --- PERBAIKAN: Logika otentikasi yang lebih jelas untuk admin dan user ---
    
    # 1. Tangani rute admin secara terpisah
    if request.path.startswith('/admin'):
        # Izinkan akses ke halaman login admin itu sendiri
        if request.endpoint == 'admin_login_page':
            return
        # Jika mencoba akses halaman admin lain tanpa sesi, alihkan ke login admin
        if 'admin_id' not in session:
            flash("Anda harus login sebagai admin untuk mengakses halaman ini.", "warning")
            return redirect(url_for('admin_login_page'))
        # Jika sudah login admin, izinkan
        return

    # 2. Tangani rute pengguna biasa
    public_routes = ['login_page', 'register_page', 'get_signal', 'static', 'status_page', 'receive_signal', 'feedback_trade', 'index', 'home_page', 'panduan_page']
    if request.endpoint in public_routes or (request.endpoint and 'static' in request.endpoint):
        return
        
    # 3. Jika rute tidak publik dan user belum login, alihkan ke login user
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

@app.route('/')
def index():
    return redirect(url_for('home_page'))

@app.route('/home')
def home_page():
    if 'user_id' in session:
        return redirect(url_for('dashboard_page'))
    return render_template('home.html', current_year=datetime.now().year)

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    form = RegisterForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        whatsapp_number = form.whatsapp_number.data
        conn = get_db()
        try:
            api_key = str(uuid.uuid4())
            today = date.today()
            end_date = today + relativedelta(days=7)
            conn.execute('''
                INSERT INTO users (username, password, api_key, start_date, end_date, status, whatsapp_number)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (username, generate_password_hash(password), api_key, today.isoformat(), end_date.isoformat(), 'trial', whatsapp_number))
            conn.commit()
            flash('Registrasi berhasil! Silakan login.', 'success')
            return redirect(url_for('login_page'))
        except Exception as e:
            flash(f'Gagal registrasi: {e}', 'danger')
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        data = request.json
        username = data.get('username')
        password = data.get('password')
        if not username or not password:
            return jsonify({"status": "error", "message": "Username dan password diperlukan"}), 400
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session.permanent = True
            return jsonify({"status": "success", "redirect_url": url_for('dashboard_page')})
        return jsonify({"status": "error", "message": "Username atau password salah"}), 401
    form = LoginForm()
    return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Anda telah logout.', 'info')
    return redirect(url_for('login_page'))

@app.route('/dashboard')
def dashboard_page():
    user_id = session.get('user_id')
    if not user_id: return redirect(url_for('login_page'))
    conn = get_db()
    user_data = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user_data: return redirect(url_for('logout'))
    status_text, badge_class = get_user_license_details(user_data)
    
    # Find the most recent signal from the available signals
    current_signal = None
    if last_signal_info:
        # Sort signals by timestamp in descending order and pick the first one
        latest_signal_key = sorted(last_signal_info, key=lambda k: last_signal_info[k]['timestamp'], reverse=True)[0]
        current_signal = last_signal_info[latest_signal_key]

    return render_template('index.html', user=user_data, license_status=status_text, badge_class=badge_class, last_signal=current_signal)

@app.route('/lisensi')
def lisensi_page():
    user_id = session.get('user_id')
    if not user_id: return redirect(url_for('login_page'))
    conn = get_db()
    user_data = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    status_text, badge_class = get_user_license_details(user_data)
    return render_template('lisensi.html', api_key=user_data['api_key'], start_date=user_data['start_date'], end_date=user_data['end_date'], license_status=status_text, badge_class=badge_class)

@app.route('/panduan')
def panduan_page():
    return render_template('panduan.html')

@app.route('/download/ea')
def download_ea():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    try:
        conn = get_db()
        user_info = conn.execute("SELECT status, end_date FROM users WHERE id = ?", (session['user_id'],)).fetchone()
        if not user_info:
            flash("Data pengguna tidak ditemukan.", "danger")
            return redirect(url_for('logout'))
        is_active = datetime.strptime(user_info['end_date'], '%Y-%m-%d').date() >= datetime.now().date() and user_info['status'] in ['trial', 'active']
        if not is_active:
            flash("Lisensi Anda tidak aktif. Silakan perpanjang lisensi untuk men-download.", 'warning')
            return redirect(url_for('dashboard_page'))
    except Exception as e:
        flash(f"Gagal memverifikasi lisensi: {e}", "danger")
        logging.error(f"Error saat verifikasi lisensi download EA untuk user {session.get('user_id')}: {e}", exc_info=True)
        return redirect(url_for('dashboard_page'))
    
    ea_directory = os.path.join(app.root_path, 'static')
    ea_filename = 'Esteh AI Update.zip' # Pastikan nama file ini sesuai
    if not os.path.exists(os.path.join(ea_directory, ea_filename)):
        flash("File EA tidak ditemukan di server.", 'danger')
        logging.error(f"File EA '{ea_filename}' tidak ditemukan di direktori: {ea_directory}")
        return redirect(url_for('dashboard_page'))
        
    logging.info(f"User {session['user_id']} berhasil men-download EA: {ea_filename}")
    return send_from_directory(directory=ea_directory, path=ea_filename, as_attachment=True)
@app.route('/subscribe')
def subscribe_page():
    form = SubscribeForm()
    return render_template('subscribe.html', form=form)

@app.route('/upload_proof', methods=['POST'])
def upload_proof():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    form = SubscribeForm()
    if form.validate_on_submit():
        file = form.proof_file.data
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{session['user_id']}_{int(time.time())}_{file.filename}")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            conn = get_db()
            conn.execute("UPDATE users SET proof_filename = ?, duration_pending = ? WHERE id = ?", (filename, form.duration.data, session['user_id']))
            conn.commit()
            flash("Bukti pembayaran berhasil diupload! Silakan tunggu aktivasi.", "success")
            return redirect(url_for('dashboard_page'))
    flash("Format file tidak valid!", "danger")
    return redirect(url_for('subscribe_page'))


@app.route('/status')
def status_page():
    api_key_to_check = INTERNAL_SECRET_KEY
    if 'user_id' in session:
        conn = get_db()
        user_data = conn.execute("SELECT api_key FROM users WHERE id = ?", (session['user_id'],)).fetchone()
        if user_data:
            api_key_to_check = user_data['api_key']
    current_signal = last_signal_info.get(api_key_to_check, last_signal_info.get(INTERNAL_SECRET_KEY))
    return render_template('status.html', last_signal=current_signal)

# === API ENDPOINTS ===
@app.route('/api/get_signal', methods=['GET'])
def get_signal():
    api_key = request.args.get('key')
    user_status = get_user_status(api_key)

    if user_status in ['invalid', 'expired', 'pending_activation']:
        return jsonify({"error": f"Unauthorized. Status: {user_status}."}), 401

    symbol = request.args.get('symbol', 'XAUUSD').upper()
    mapped_symbol = SYMBOL_ALIAS_MAP.get(symbol, symbol)
    # The key for last_signal_info should be consistent for all users
    signal_data_key = f"INTERNAL_SIGNAL_{mapped_symbol}"
    signal_data = last_signal_info.get(signal_data_key)

    if not signal_data or (datetime.now() - datetime.strptime(signal_data['timestamp'], '%Y-%m-%d %H:%M:%S')).total_seconds() >= 300:
        return jsonify({"order_type": "WAIT", "reason": "No fresh signal from server."})

    if user_status == 'trial':
        trial_settings = current_app.config.get('APP_CONFIG', {}).get('trial_user_settings', {})
        signal_id = signal_data.get('signal_id')
        context = signal_context_cache.get(signal_id, {})
        profile_name = context.get('profile_name')
        
        allowed_profiles = trial_settings.get('allowed_profiles', [])
        if profile_name and profile_name not in allowed_profiles:
            return jsonify({"order_type": "WAIT", "reason": f"Profile '{profile_name}' requires premium subscription."})

        today_str = date.today().isoformat()
        user_counts = daily_signal_counts.get(api_key, {'date': today_str, 'count': 0})
        if user_counts['date'] != today_str:
            user_counts = {'date': today_str, 'count': 0}
        
        limit = trial_settings.get('daily_signal_limit', 3)
        if user_counts['count'] >= limit:
            return jsonify({"order_type": "WAIT", "reason": f"Daily signal limit ({limit}) reached."})
        
        logging.info(f"Trial user {api_key} consuming signal {user_counts.get('count', 0) + 1}/{limit} for today.")
        daily_signal_counts[api_key] = {'date': today_str, 'count': user_counts.get('count', 0) + 1}

    response_data = signal_data['signal_json'].copy()
    response_data.update({"signal_id": signal_data['signal_id'], "order_type": signal_data['order_type']})
    return jsonify(response_data)

@app.route('/api/internal/submit_signal', methods=['POST'])
def receive_signal():
    global last_signal_info, signal_context_cache
    data = request.json
    if not data: return jsonify({"error": "No data received"}), 400
    
    # --- PERUBAHAN: Validasi secret key internal ---
    if data.get('secret_key') != INTERNAL_SECRET_KEY:
        logging.warning("Upaya submit sinyal dengan secret key yang salah ditolak.")
        return jsonify({"error": "Invalid secret key"}), 401

    api_key = data.get('api_key')
    if not api_key:
        logging.warning("Upaya submit sinyal tanpa API key ditolak.")
        return jsonify({"error": "API key is required"}), 400

    symbol = data.get('symbol', 'XAUUSD').upper()
    mapped_symbol = SYMBOL_ALIAS_MAP.get(symbol, symbol)
    order_type = data.get('order_type')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    signal_id = generate_signal_id(api_key, order_type, timestamp)

    signal_context_cache[signal_id] = {
        "score": data.get('score'), "info": data.get('info'), "timestamp": timestamp,
        "symbol": symbol, "order_type": order_type, "profile_name": data.get('profile_name'),
        "score_components": data.get('score_components')
    }
    
    signal_payload = {
        'signal_id': signal_id, 'order_type': order_type,
        'timestamp': timestamp, 'signal_json': data.get('signal_json', {})
    }
    
    # The key for last_signal_info should be consistent for all users
    key_to_update = f"INTERNAL_SIGNAL_{mapped_symbol}"
    last_signal_info[key_to_update] = signal_payload
    logging.info(f"📢 Sinyal BARU diterima & disiarkan: Type={order_type}, Simbol={symbol}.")
    
    # --- PENGEMBANGAN: Kirim notifikasi real-time ke dashboard ---
    notification_data = {
        "symbol": symbol,
        "order_type": order_type,
        "profile_name": data.get('profile_name'),
        "score": data.get('score'),
        "entry_price": data.get('signal_json', {}).get(f"{order_type.capitalize()}Entry", "N/A")
    }
    socketio.emit('new_signal_notification', notification_data)
    
    return jsonify({"message": "Signal received"}), 200


@app.route('/api/feedback_trade', methods=['POST'])
def feedback_trade():
    data = request.json
    if not data:
        app.logger.error("No data received in feedback_trade")
        return jsonify({"status": "error", "message": "No data received"}), 400
    
    signal_id = data.get('signal_id')
    if not signal_id:
        app.logger.error("No signal_id provided")
        return jsonify({"status": "error", "message": "No signal_id provided"}), 400
    
    context = signal_context_cache.pop(signal_id, {})
    full_feedback = {**context, **data}
    full_feedback['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    feedback_path = "trade_feedback.json"
    with feedback_file_lock:
        try:
            with open(feedback_path, "r+") as f:
                current_data = json.load(f)
                current_data.append(full_feedback)
                f.seek(0)
                json.dump(current_data, f, indent=2)
            app.logger.info(f"Feedback saved for signal_id {signal_id}: {full_feedback['result']}")
        except (FileNotFoundError, json.JSONDecodeError):
            with open(feedback_path, "w") as f:
                json.dump([full_feedback], f, indent=2)
            app.logger.info(f"New feedback file created with signal_id {signal_id}")
        except Exception as e:
            app.logger.error(f"Error saving feedback: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    # Auto-adaptation
    with open(feedback_path, "r") as f:
        trades = json.load(f)
    valid_trades = [t for t in trades if t.get('result') in ['win', 'loss'] and t.get('profile_name') == full_feedback.get('profile_name')]
    if len(valid_trades) >= 50:
        wins = sum(1 for t in valid_trades if t['result'] == 'win')
        win_rate = wins / len(valid_trades) if valid_trades else 0
        if win_rate < 0.55:
            loss_trades = [t for t in valid_trades if t['result'] == 'loss']
            components = [comp for trade in loss_trades for comp in trade.get('score_components', [])]
            common_components = Counter(components).most_common(1)
            if common_components:
                problematic_comp = common_components[0][0]
                adjustment = -0.1
                app.logger.info(f"Auto-adaptation: Kurangi weight untuk {problematic_comp} sebesar {adjustment}")
                try:
                    with open("config.json", "r+") as f:
                        config = json.load(f)
                        for symbol in config["symbols"]:
                            if problematic_comp in config["symbols"][symbol]["weights"]:
                                config["symbols"][symbol]["weights"][problematic_comp] += adjustment
                        f.seek(0)
                        json.dump(config, f, indent=2)
                    app.logger.info(f"Config updated: {problematic_comp} weight adjusted")
                except Exception as e:
                    app.logger.error(f"Gagal update config.json: {e}")
    
    return jsonify({"status": "success"}), 200

# === ADMIN PANEL ROUTES ===
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_id' not in session:
            flash("Anda harus login sebagai admin untuk mengakses halaman ini.", "warning")
            return redirect(url_for('admin_login_page'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login_page():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        conn = get_db()
        admin = conn.execute("SELECT * FROM admins WHERE username = ?", (username,)).fetchone()
        if admin and check_password_hash(admin['password'], password):
            session['admin_id'] = admin['id']
            flash('Login admin berhasil!', 'success')
            return redirect(url_for('admin_dashboard_page'))
        else:
            flash('Username atau password admin salah.', 'danger')
    return render_template('admin_login.html', form=form)

@app.route('/admin/logout')
@admin_required
def admin_logout():
    session.pop('admin_id', None)
    flash('Anda telah logout dari panel admin.', 'info')
    return redirect(url_for('admin_login_page'))

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard_page():
    conn = get_db()
    users = conn.execute("SELECT id, username, start_date, end_date, status, proof_filename FROM users ORDER BY id").fetchall()
    return render_template('admin_dashboard.html', users=users)

@app.route('/admin/extend_license', methods=['POST'])
@admin_required
def extend_license():
    user_id = request.form.get('user_id')
    days_to_add = int(request.form.get('days', 30))
    
    conn = get_db()
    user = conn.execute("SELECT end_date, status FROM users WHERE id = ?", (user_id,)).fetchone()
    
    if user:
        current_end_date = datetime.strptime(user['end_date'], '%Y-%m-%d').date()
        # Jika lisensi sudah kadaluarsa, perpanjang dari hari ini
        if current_end_date < date.today():
            new_end_date = date.today() + timedelta(days=days_to_add)
        else:
            new_end_date = current_end_date + timedelta(days=days_to_add)
        
        # Jika statusnya pending, ubah jadi active
        new_status = 'active'
        
        conn.execute(
            "UPDATE users SET end_date = ?, status = ?, duration_pending = NULL, proof_filename = NULL WHERE id = ?",
            (new_end_date.isoformat(), new_status, user_id)
        )
        conn.commit()
        flash(f"Lisensi untuk user ID {user_id} berhasil diperpanjang selama {days_to_add} hari.", "success")
    else:
        flash(f"User ID {user_id} tidak ditemukan.", "danger")
        
    return redirect(url_for('admin_dashboard_page'))


# === SOCKET.IO EVENTS ===
@socketio.on('connect')
def handle_connect():
    logging.info(f"Browser client terhubung: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    logging.info(f"Browser client terputus: {request.sid}")

@socketio.on('submit_log')
def handle_submit_log(data):
    socketio.emit('new_log', {'message': data.get('message', '')})

# ========== INIT & RUN ==========
def load_app_config():
    try:
        with open('config.json', 'r') as f:
            app.config['APP_CONFIG'] = json.load(f)
        logging.info("Konfigurasi aplikasi dimuat dari config.json")
    except Exception as e:
        logging.error(f"FATAL: Tidak dapat memuat config.json: {e}")
        app.config['APP_CONFIG'] = {}

if __name__ == '__main__':
    os.makedirs(os.path.join(app.root_path, 'static'), exist_ok=True)
    os.makedirs(os.path.join(app.root_path, app.config['UPLOAD_FOLDER']), exist_ok=True)
    with app.app_context():
        init_db_data()
        load_app_config()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)