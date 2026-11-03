from flask import Flask, render_template, request, redirect, url_for, session, flash
import requests
from flask_wtf.csrf import CSRFProtect, CSRFError
from werkzeug.security import generate_password_hash, check_password_hash
from database.db import get_connection
from config import Config
from plans import build_diet_plan, build_exercise_plan
from explainability.shap_explainer import explain_prediction
from datetime import datetime, timedelta
import pandas as pd
import joblib
import os
import random
import secrets
import re
from urllib.parse import quote_plus

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY

# CSRF protection for every POST form (each form carries a hidden csrf_token)
csrf = CSRFProtect(app)


@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    flash("Your session expired or the form was invalid. Please try again.", "danger")
    return redirect(url_for("login_page"))

# ---------------- EMAIL CONFIGURATION (Resend HTTP API) ----------------
# Set these as environment variables on your hosting platform:
#   RESEND_API_KEY = re_xxxxxxxx
#   RESEND_FROM    = PCOS Care <noreply@yourdomain.com>   (a verified domain)
# Until a domain is verified, Resend only lets "onboarding@resend.dev" send,
# and only to the email address of your own Resend account.

RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_FROM = os.getenv("RESEND_FROM", "PCOS Care <onboarding@resend.dev>")

print("Resend API key loaded:", bool(RESEND_API_KEY))
print("Resend sender:", RESEND_FROM)


def send_email(to, subject, body):
    """Send a plain-text email through Resend. Raises RuntimeError on failure."""
    if not RESEND_API_KEY:
        raise RuntimeError("RESEND_API_KEY is not set")

    resp = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"from": RESEND_FROM, "to": [to], "subject": subject, "text": body},
        timeout=15,
    )

    if resp.status_code >= 300:
        raise RuntimeError(f"Resend error {resp.status_code}: {resp.text}")


# ---------------- DATABASE CONNECTION TEST ----------------

try:
    conn = get_connection()
    print("✅ Connected to MySQL Successfully!")
    conn.close()
except Exception as e:
    print("❌ Database Connection Error:", e)

# ---------------- LOAD MODEL ----------------

# ==========================================================
# LOAD AI MODELS
# ==========================================================

if os.path.exists("random_forest.joblib"):
    rf_model = joblib.load("random_forest.joblib")
    print("✅ Random Forest model loaded!")
else:
    rf_model = None
    print("❌ random_forest.joblib not found!")

if os.path.exists("xgboost.joblib"):
    xgb_model = joblib.load("xgboost.joblib")
    print("✅ XGBoost model loaded!")
else:
    xgb_model = None
    print("❌ xgboost.joblib not found!")

if os.path.exists("pcos_pipeline.joblib"):
    best_model = joblib.load("pcos_pipeline.joblib")
    print("✅ Best model loaded!")
else:
    best_model = None
    print("❌ pcos_pipeline.joblib not found!")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# ==========================================================
# LOGIN PAGE
# ==========================================================

@app.route('/')
def landing():

    # Public landing page; logged-in users see a Dashboard button instead of Login
    return render_template("landing.html")


@app.route('/login')
def login_page():

    if 'user_id' in session:
        return redirect(url_for('home'))

    return render_template("login.html")


# ==========================================================
# LOGIN
# ==========================================================

@app.route('/login_action', methods=['POST'])
def login_action():

    email = request.form['email'].strip().lower()
    password = request.form['password']

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE email=%s",
        (email,)
    )

    user = cursor.fetchone()

    conn.close()

    if user and check_password_hash(user['password'], password):

        session['user_id'] = user['id']
        session['user_name'] = user['full_name']
        session['user_email'] = user['email']
        session['role'] = user.get('role', 'user')
        flash(f"Welcome, {user['full_name']}!", "success")
        return redirect(url_for('home'))

    flash("Invalid Email or Password.", "danger")
    return redirect(url_for("login_page"))

# ==========================================================
# REGISTER
# ==========================================================

@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == "POST":

        full_name = request.form['full_name'].strip()
        age = request.form['age']
        email = request.form['email'].strip().lower()
        raw_password = request.form['password']

        error = None
        if not full_name:
            error = "Please enter your name."
        elif not EMAIL_RE.match(email):
            error = "Please enter a valid email address."
        elif not age.isdigit() or not (10 <= int(age) <= 100):
            error = "Please enter a valid age (10-100)."
        elif len(raw_password) < 8:
            error = "Password must be at least 8 characters."

        if error:
            return render_template("register.html", error=error)

        password = generate_password_hash(raw_password)

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE email=%s",
            (email,)
        )

        existing = cursor.fetchone()

        if existing:

            conn.close()

            return render_template(
                "register.html",
                error="Email already registered!"
            )

        cursor.execute(
            """
            INSERT INTO users
            (full_name, age, email, password)
            VALUES (%s,%s,%s,%s)
            """,
            (full_name, age, email, password)
        )

        conn.commit()
        conn.close()
        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login_page'))

    return render_template("register.html")

# ==========================================================
# HOME
# ==========================================================

@app.route('/home')
def home():

    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    return render_template("index.html")


# ==========================================================
# PREDICTION
# ==========================================================
@app.route('/predict', methods=['POST'])
def predict():

    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    try:
        # Name is only for display on the report; it is NOT a model feature
        patient_name = request.form.get('Name', '').strip()
        if not patient_name:
            raise ValueError("Please enter your full name.")
        if len(patient_name) > 120:
            raise ValueError("Name must be 120 characters or fewer.")

        age = int(request.form.get('Age'))
        bmi = float(request.form.get('BMI'))
        cycle_length = int(request.form.get('Cycle_Length'))
        hirsutism = int(request.form.get('Hirsutism_Score'))

        if not (10 <= age <= 100):
            raise ValueError("Age must be between 10 and 100.")
        if not (10 <= bmi <= 70):
            raise ValueError("BMI must be between 10 and 70.")
        if not (15 <= cycle_length <= 120):
            raise ValueError("Cycle length must be between 15 and 120 days.")
        if hirsutism not in (0, 1, 2, 3):
            raise ValueError("Invalid excess-hair score.")

        # Model input: exactly the features the model was trained on (no Name)
        input_data = {
            'Country': [request.form.get('Country')],
            'Age': [age],
            'BMI': [bmi],
            'Cycle_Length': [cycle_length],
            'Cycle_Regularity': [request.form.get('Cycle_Regularity')],
            'Hirsutism_Score': [hirsutism],
            'Hyperandrogenism': [request.form.get('Hyperandrogenism')],
            'Acne': [request.form.get('Acne')],
            'Hair_Thinning': [request.form.get('Hair_Thinning')],
            'Skin_Darkening': [request.form.get('Skin_Darkening')],
            'Rapid_Weight_Gain': [request.form.get('Rapid_Weight_Gain')],
            'Stress_Level': [request.form.get('Stress_Level')],
            'Lifestyle_Score': [request.form.get('Lifestyle_Score')]
        }

        df = pd.DataFrame(input_data)
        top_features = explain_prediction(df)
        rf_prediction = rf_model.predict(df)[0]
        xgb_prediction = xgb_model.predict(df)[0]
        prediction = best_model.predict(df)[0]

        rf_probability = rf_model.predict_proba(df)[0][1] * 100
        xgb_probability = xgb_model.predict_proba(df)[0][1] * 100

        # The final result, risk % and label all come from the SAME best model
        best_probability = best_model.predict_proba(df)[0][1] * 100
        selected_model = (
            "XGBoost"
            if best_model.named_steps["classifier"].__class__.__name__ == "XGBClassifier"
            else "Random Forest"
        )

        risk_percentage = round(best_probability, 2)

        if prediction == 1:
            result = "High Risk Detected"
            color = "red"
        else:
            result = "Low Risk Profile"
            color = "green"

        # Save prediction to database
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO prediction_history
            (user_id, prediction, bmi, cycle_length, risk_percentage)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            session['user_id'],
            result,
            float(df['BMI'][0]),
            int(df['Cycle_Length'][0]),
            risk_percentage
        ))

        conn.commit()
        conn.close()

        return render_template(
            'result.html',
            result=result,
            color=color,
            patient_name=patient_name,
            user_data=request.form,
            bmi_val=df['BMI'][0],

            rf_prediction="High Risk" if rf_prediction == 1 else "Low Risk",
            xgb_prediction="High Risk" if xgb_prediction == 1 else "Low Risk",

            rf_probability=round(rf_probability, 2),
            xgb_probability=round(xgb_probability, 2),

            selected_model=selected_model,
            risk_percentage=round(risk_percentage, 2),
            top_features=top_features
        )

    except ValueError as e:
        flash(str(e), "danger")
        return redirect(url_for("home"))

    except Exception:
        app.logger.exception("Prediction failed")
        flash("Something went wrong while making the prediction. Please check your inputs and try again.", "danger")
        return redirect(url_for("home"))

# ==========================================================
# PREDICTION HISTORY
# ==========================================================

@app.route('/history')
def history():

    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM prediction_history
        WHERE user_id=%s
        ORDER BY prediction_date DESC
    """, (session['user_id'],))

    history = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        history=history,
        user_name=session['user_name']
    )

# ==========================================================
# LOGOUT
# ==========================================================

@app.route('/logout')
def logout():

    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for('login_page'))

# ==========================================================
# CYCLE TRACKER PAGE
# ==========================================================

@app.route("/cycle_tracker")
def cycle_tracker():

    if "user_id" not in session:
        return redirect(url_for("login_page"))

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT period_start,
               cycle_length,
               next_period,
               created_at
        FROM cycle_tracker
        WHERE user_id = %s
        ORDER BY created_at DESC
    """, (session["user_id"],))

    history = cursor.fetchall()

    conn.close()

    return render_template(
        "cycle_tracker.html",
        history=history
    )


# ==========================================================
# SAVE CYCLE
# ==========================================================

@app.route("/save_cycle", methods=["POST"])
def save_cycle():

    if "user_id" not in session:
        return redirect(url_for("login_page"))

    start_date = request.form["period_start"]
    cycle_length = int(request.form["cycle_length"])

    next_period = datetime.strptime(
        start_date,
        "%Y-%m-%d"
    ) + timedelta(days=cycle_length)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO cycle_tracker
        (user_id, period_start, cycle_length, next_period)
        VALUES(%s,%s,%s,%s)
    """, (
        session["user_id"],
        start_date,
        cycle_length,
        next_period.date()
    ))

    conn.commit()
    conn.close()

    return redirect(url_for("cycle_tracker"))

def get_latest_prediction(user_id):
    """Most recent screening for this user (or None)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT prediction, bmi
        FROM prediction_history
        WHERE user_id=%s
        ORDER BY prediction_date DESC
        LIMIT 1
    """, (user_id,))
    row = cursor.fetchone()
    conn.close()
    return row


def parse_bmi(value):
    bmi = float(value)
    if not (10 <= bmi <= 70):
        raise ValueError("BMI must be between 10 and 70.")
    return bmi


@app.route("/diet_planner")
def diet_planner():

    if "user_id" not in session:
        return redirect(url_for("login_page"))

    latest = get_latest_prediction(session["user_id"])

    return render_template(
        "diet_planner.html",
        prefill_bmi=latest["bmi"] if latest else None,
        high_risk=bool(latest and "High" in latest["prediction"])
    )


@app.route("/generate_diet", methods=["POST"])
def generate_diet():

    if "user_id" not in session:
        return redirect(url_for("login_page"))

    try:
        bmi = parse_bmi(request.form["bmi"])
    except (ValueError, KeyError):
        flash("Please enter a valid BMI between 10 and 70.", "danger")
        return redirect(url_for("diet_planner"))

    preference = request.form.get("preference", "Vegetarian")
    insulin_issue = request.form.get("insulin_issue") == "yes"

    latest = get_latest_prediction(session["user_id"])
    high_risk = bool(latest and "High" in latest["prediction"])

    plan = build_diet_plan(bmi, preference, insulin_issue, high_risk)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO diet_plans
        (user_id,risk_level,bmi,breakfast,lunch,snacks,dinner)
        VALUES(%s,%s,%s,%s,%s,%s,%s)
    """, (
        session["user_id"],
        plan["db_level"],
        bmi,
        plan["breakfast"][:250],
        plan["lunch"][:250],
        plan["snacks"][:250],
        plan["dinner"][:250]
    ))

    conn.commit()
    conn.close()

    return render_template(
        "diet_planner.html",
        plan=plan,
        bmi=bmi,
        prefill_bmi=bmi,
        high_risk=high_risk,
        form_pref=plan["preference"],
        form_insulin=insulin_issue
    )


@app.route("/exercise_planner")
def exercise_planner():

    if "user_id" not in session:
        return redirect(url_for("login_page"))

    latest = get_latest_prediction(session["user_id"])

    return render_template(
        "exercise_planner.html",
        prefill_bmi=latest["bmi"] if latest else None,
        high_risk=bool(latest and "High" in latest["prediction"])
    )


@app.route("/generate_exercise", methods=["POST"])
def generate_exercise():

    if "user_id" not in session:
        return redirect(url_for("login_page"))

    try:
        bmi = parse_bmi(request.form["bmi"])
    except (ValueError, KeyError):
        flash("Please enter a valid BMI between 10 and 70.", "danger")
        return redirect(url_for("exercise_planner"))

    level = request.form.get("level", "Beginner")
    joint_issue = request.form.get("joint_issue") == "yes"

    latest = get_latest_prediction(session["user_id"])
    high_risk = bool(latest and "High" in latest["prediction"])

    plan = build_exercise_plan(bmi, level, joint_issue, high_risk)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO exercise_plans
        (user_id,risk_level,bmi,warmup,cardio,strength_training,yoga,duration)
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        session["user_id"],
        plan["db_level"],
        bmi,
        plan["warmup"][:250],
        plan["cardio"][:250],
        plan["strength"][:250],
        plan["yoga"][:250],
        plan["duration"]
    ))

    conn.commit()
    conn.close()

    return render_template(
        "exercise_planner.html",
        plan=plan,
        bmi=bmi,
        prefill_bmi=bmi,
        high_risk=high_risk,
        form_level=plan["level"],
        form_joint=joint_issue
    )

@app.route("/admin")
def admin():

    if "user_id" not in session:
        return redirect(url_for("login_page"))

    # Only these users can access the Admin Dashboard
    allowed_admins = [
        "syedaasra2407@gmail.com",
        "khushigupta46576@gmail.com",
        "ayushikashyap91427@gmail.com"
    ]

    if session.get("user_email") not in allowed_admins:
        return redirect(url_for("home"))

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) AS total FROM users")
    total_users = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) AS total FROM prediction_history")
    total_predictions = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) AS total FROM cycle_tracker")
    total_cycles = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) AS total FROM diet_plans")
    total_diets = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) AS total FROM exercise_plans")
    total_exercises = cursor.fetchone()["total"]

    cursor.execute("""
    SELECT *
    FROM prediction_history
    ORDER BY prediction_date DESC
    LIMIT 10
    """)

    recent_predictions = cursor.fetchall()

    conn.close()

    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        total_predictions=total_predictions,
        total_cycles=total_cycles,
        total_diets=total_diets,
        total_exercises=total_exercises,
        recent_predictions=recent_predictions
    )

@app.route("/forgot_password")
def forgot_password():
    return render_template("forgot_password.html")

@app.route("/send_otp", methods=["POST"])
def send_otp():

    email = request.form["email"]

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE email=%s",
        (email,)
    )

    user = cursor.fetchone()

    if not user:
        conn.close()
        return render_template(
            "forgot_password.html",
            error="Email not registered."
        )

    session.pop("otp_attempts", None)
    otp = str(secrets.randbelow(900000) + 100000)

    # Remove any older OTPs for this email so only the newest works
    cursor.execute(
        "DELETE FROM password_reset_otp WHERE email=%s",
        (email,)
    )

    cursor.execute(
        """
        INSERT INTO password_reset_otp(email, otp)
        VALUES(%s,%s)
        """,
        (email, otp)
    )

    conn.commit()
    conn.close()

    try:
        send_email(
            email,
            "PCOS Care Password Reset OTP",
            f"Your OTP is: {otp}\n\nIt is valid for 10 minutes. If you did not request this, ignore this email."
        )
    except Exception:
        app.logger.exception("Could not send OTP email")
        return render_template(
            "forgot_password.html",
            error="We could not send the email right now. Please try again in a few minutes."
        )

    return render_template(
        "verify_otp.html",
        email=email
    )


@app.route("/reset_password", methods=["POST"])
def reset_password():

    email = session.get("reset_email")

    # Only allowed after the OTP for this email was verified
    if not email or email != request.form.get("email"):
        flash("Please verify your OTP first.", "danger")
        return redirect(url_for("forgot_password"))

    new_password = request.form["password"]

    if len(new_password) < 8:
        return render_template(
            "reset_password.html",
            email=email,
            error="Password must be at least 8 characters."
        )

    password = generate_password_hash(new_password)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE users
        SET password=%s
        WHERE email=%s
        """,
        (password, email)
    )

    conn.commit()
    conn.close()

    session.pop("reset_email", None)

    flash("Password changed successfully. Please login.", "success")
    return redirect(url_for("login_page"))

@app.route("/doctor_search", methods=["POST"])
def doctor_search():

    city = request.form["city"].strip()[:100]

    maps_url = "https://www.google.com/maps/search/gynecologist+in+" + quote_plus(city)

    return redirect(maps_url)

@app.route("/education")
def education():

    if "user_id" not in session:
        return redirect(url_for("login_page"))

    return render_template("education.html")

@app.route("/email_report", methods=["POST"])
def email_report():

    if "user_id" not in session:
        return redirect(url_for("login_page"))

    email = session["user_email"]

    # Use the saved result from the database, never values sent by the browser
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT prediction, bmi, risk_percentage
        FROM prediction_history
        WHERE user_id=%s
        ORDER BY prediction_date DESC
        LIMIT 1
    """, (session["user_id"],))
    latest = cursor.fetchone()
    conn.close()

    if not latest:
        flash("Please run a screening first, then email the report.", "danger")
        return redirect(url_for("home"))

    result = latest["prediction"]
    risk = latest["risk_percentage"]
    bmi = latest["bmi"]

    body = f"""
PCOS CARE - AI HEALTH REPORT

Prediction : {result}

Risk Percentage : {risk}%

BMI : {bmi}

Thank you for using PCOS Care.

This report is generated using Artificial Intelligence and should not replace professional medical advice.

Regards,
PCOS Care Team
"""

    try:
        send_email(email, "Your PCOS Care AI Health Report", body)
    except Exception:
        app.logger.exception("Could not send report email")
        flash("We could not send the report email right now. Please try again later.", "danger")
        return redirect(url_for("home"))

    flash("Health report sent successfully!", "success")
    return render_template(
        "email_success.html",
        email=email
    )

@app.route("/profile")
def profile():

    if "user_id" not in session:
        return redirect(url_for("login_page"))

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE id=%s",
        (session["user_id"],)
    )

    user = cursor.fetchone()

    conn.close()

    return render_template(
        "profile.html",
        user=user
    )


@app.route("/update_profile", methods=["POST"])
def update_profile():

    if "user_id" not in session:
        return redirect(url_for("login_page"))

    full_name = request.form["name"].strip()
    email = request.form["email"].strip().lower()

    if not full_name or not EMAIL_RE.match(email):
        flash("Please enter a valid name and email.", "danger")
        return redirect(url_for("profile"))

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id FROM users WHERE email=%s AND id<>%s",
        (email, session["user_id"])
    )

    if cursor.fetchone():
        conn.close()
        flash("That email is already used by another account.", "danger")
        return redirect(url_for("profile"))

    cursor.execute("""
        UPDATE users
        SET full_name=%s,
            email=%s
        WHERE id=%s
    """, (
        full_name,
        email,
        session["user_id"]
    ))

    conn.commit()
    conn.close()

    # Update session values
    session["user_name"] = full_name
    session["user_email"] = email
    flash("Profile updated successfully.", "success")
    return redirect(url_for("profile"))

@app.route("/settings")
def settings():

    if "user_id" not in session:
        return redirect(url_for("login_page"))

    return render_template("settings.html")

@app.route("/doctor_finder")
def doctor_finder():

    if "user_id" not in session:
        return redirect(url_for("login_page"))

    return render_template("doctor_finder.html")

@app.route("/verify_otp", methods=["POST"])
def verify_otp():

    email = request.form.get("email")
    otp = request.form.get("otp")

    if not email or not otp:
        return render_template(
            "verify_otp.html",
            email=email,
            error="Please enter the OTP."
        )

    attempts = session.get("otp_attempts", 0)

    if attempts >= 5:
        return render_template(
            "forgot_password.html",
            error="Too many wrong attempts. Please request a new OTP."
        )

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT otp,
               TIMESTAMPDIFF(MINUTE, created_at, NOW()) AS age_minutes
        FROM password_reset_otp
        WHERE email=%s
        ORDER BY created_at DESC
        LIMIT 1
    """, (email,))

    record = cursor.fetchone()

    if record is None:
        conn.close()
        return render_template(
            "forgot_password.html",
            error="OTP not found. Please request a new OTP."
        )

    if record["age_minutes"] >= 10:
        cursor.execute("DELETE FROM password_reset_otp WHERE email=%s", (email,))
        conn.commit()
        conn.close()
        return render_template(
            "forgot_password.html",
            error="OTP expired. Please request a new OTP."
        )

    if secrets.compare_digest(record["otp"], otp.strip()):

        # One-time use: delete it and remember this email is verified
        cursor.execute("DELETE FROM password_reset_otp WHERE email=%s", (email,))
        conn.commit()
        conn.close()

        session["reset_email"] = email
        session.pop("otp_attempts", None)

        return render_template(
            "reset_password.html",
            email=email
        )

    conn.close()
    session["otp_attempts"] = attempts + 1

    return render_template(
        "verify_otp.html",
        email=email,
        error="Invalid OTP. Please try again."
    )

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1")
