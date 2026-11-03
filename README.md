# 🌸 PCOS Care
<h3 align="center">AI-Powered PCOS Risk Prediction & Women's Wellness Platform</h3>

<p align="center">
  A machine-learning-powered web application designed to support PCOS risk awareness and women's wellness tracking.
</p>

<p align="center">
  <a href="https://pcos-care-ybgh.onrender.com/home">
    <img src="https://img.shields.io/badge/🌐_Live_Demo-Visit_PCOS_Care-8B5CF6?style=for-the-badge" alt="Live Demo"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white" alt="MySQL"/>
</p>

---

## ✨ About The Project

**PCOS Care** is a web-based healthcare support platform that uses machine learning to estimate PCOS risk from relevant health and lifestyle information.

The project combines predictive analytics with wellness-oriented features to help users explore their health information through an accessible web interface.

**Project Goals**
- Apply machine learning to a women's health use case.
- Build an end-to-end application integrating a trained model with a web interface.
- Provide health tracking and wellness-support functionality.
- Demonstrate the practical application of Python, data processing and predictive modelling.

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| 🧠 **PCOS Risk Prediction** | Estimate PCOS risk using the trained machine learning pipeline. |
| 🔐 **User Authentication** | User registration, login and email OTP verification. |
| 📊 **Health Dashboard** | Review relevant recorded health and progress information. |
| 🗓️ **Menstrual Cycle Tracker** | Record and monitor menstrual cycle information. |
| 🥗 **Diet Planner** | Access wellness-oriented dietary guidance. |
| 🏃 **Exercise Planner** | Explore exercise and lifestyle guidance. |
| 📈 **Progress Tracking** | Review recorded information over time. |
| 📄 **PDF Report** | Generate a report if the feature is enabled in your application. |
| 🔎 **Explainable AI** | SHAP-based model interpretation, if implemented. |

*Keep only features that are implemented in your actual project.*

---

## 🛠️ Technology Stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/XGBoost-189B52?style=flat-square" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/MySQL-4479A1?style=flat-square&logo=mysql&logoColor=white" alt="MySQL"/>
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white" alt="HTML5"/>
  <img src="https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white" alt="CSS3"/>
  <img src="https://img.shields.io/badge/Render-46E3B7?style=flat-square&logo=render&logoColor=black" alt="Render"/>
</p>

- **Language:** Python
- **Backend:** Flask
- **Machine Learning:** XGBoost, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Database:** MySQL
- **Frontend:** HTML, CSS, JavaScript and Jinja2 templates, where used
- **Deployment:** Render
- **Model Persistence:** Joblib

---

## 📸 Application Preview

Add screenshots of your actual application below. Upload them into a folder named `screenshots` in your repository.

| 🏠 Home Page | 🔐 Login Page |
|---|---|
| ![PCOS Care Home](screenshots/home.png) | ![Login Page](screenshots/login.png) |

| 🧠 Prediction Results | 📊 Dashboard |
|---|---|
| ![Prediction Results](screenshots/prediction.png) | ![Dashboard](screenshots/dashboard.png) |

> Replace the example filenames with the actual names of your screenshots.

---

## ⚙️ How It Works

1. **Input:** The user enters the health and lifestyle information requested by the application.
2. **Preprocessing:** The application prepares the input for the trained model.
3. **Prediction:** The machine learning pipeline generates a PCOS risk estimate.
4. **Results:** The application displays the prediction through the web interface.
5. **Tracking:** Available dashboard features allow users to review recorded information.

---

## 💻 Run Locally

### Prerequisites

- Python
- MySQL Server
- Git

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_FOLDER
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Set up the environment variables required by the application, including database connection details, Flask secret key and email configuration.

**Never upload passwords, secret keys or your `.env` file to GitHub.**

### 5. Start the Application

```bash
python app.py
```

Follow the local URL printed in your terminal.

---

## 🌐 Live Application

<p align="center">
  <a href="https://pcos-care-ybgh.onrender.com/home">
    <img src="https://img.shields.io/badge/OPEN_LIVE_APPLICATION-8B5CF6?style=for-the-badge&logo=render&logoColor=white" alt="Open Live Application"/>
  </a>
</p>

**Live URL:** https://pcos-care-ybgh.onrender.com

---

## 🎯 Learning Outcomes

Through this project, I explored:

- Integrating a trained machine learning model into a Flask application.
- Preparing data and working with a predictive modelling pipeline.
- Connecting a web application to a relational database.
- Implementing authentication and email OTP workflows.
- Deploying a Python application using Render.

---

## ⚠️ Medical Disclaimer

PCOS Care is an educational project intended to demonstrate
