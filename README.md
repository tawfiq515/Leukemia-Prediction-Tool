### Leukemia Prediction Tool (Streamlit + ML)

Developed by [Tawfiq](https://github.com/tawfiq515) & [Rama Al Jada’](https://github.com/RamaAljada)


Interactive Streamlit web app powered by Machine Learning to estimate leukemia risk from blood test inputs.  
Supports **CSV upload** and **manual input**, auto-maps messy column names via a smart alias dictionary, and saves per-user history to compare trends over time.

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

---

## Features
- Two input modes: **CSV upload** or **Manual input**.
- Models: **Random Forest**, **XGBoost**, **Logistic Regression** (+ **Regression** for risk %).
- Clustering: **KMeans**, **DBSCAN**, **Agglomerative**.
- **Alias dictionary** auto-maps lab names (e.g., `White Blood Cells`, `w.b.c` → `WBC`).
- **User login-like name** → saves results to `history_<username>.csv` and compares with previous runs.
- Custom background + styled UI (blue bold text).


---

## Quickstart

Follow these steps to set up and run the app locally.

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Leukemia-Prediction-Tool.git
cd Leukemia-Prediction-Tool
```

### 2. Run the app
Streamlit will:

Start a local web server.

Open http://localhost:8501 in your browser automatically.

If the browser doesn’t open, copy the URL from the terminal.

### Try it out

Select Upload CSV → upload a blood test file.

Or choose Manual Input → type values directly.

Click Analyze → see classification results, risk %, recommendations, and history comparison.

### Stop the server

Press Ctrl + C in the terminal to exit.


### 3. Data Format (CSV)

The app supports multiple aliases for common blood test names, thanks to the smart dictionary.

Standard Column	Accepted Aliases (examples)	Type	Default
WBC	WBC, White Blood Cells, white_blood_cells, w.b.c	float	7.0
RBC	RBC, Red Blood Cells, red_blood_cells, r.b.c	float	4.5
Hb	Hb, HGB, Hemoglobin, Haemoglobin	float	13.5
Platelets	Platelets, PLT, platelet	float	250.0
Lymphocytes	Lymphocytes, lymphs, lym	float	30.0

Sample CSV:

WBC,RBC,Hb,Platelets,Lymphocytes
8.1,4.7,14.2,270,32
6.3,4.4,13.1,240,28

With mixed names (auto-mapped):

White Blood Cells,red_blood_cells,HGB,PLT,lymphs
9.5,4.3,12.8,220,35


### 4. Deploy to Streamlit Cloud
``` [theme]
base="light"
primaryColor="#0077B6"
textColor="#0077B6"
```


### 4. Acknowledgments

Built by 
### Tawfiq and Rama Al Jada’.
Special thanks to open-source libraries: Streamlit, scikit-learn, XGBoost, Pandas, and NumPy.



