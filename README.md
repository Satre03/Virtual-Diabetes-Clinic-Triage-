# Virtual-Diabetes-Clinic-Triage-

# Diabetes Risk Service

This project is a small ML-service designed to predict shortterm diabetes progression by using the open *scikit-learn Diabetes* dataset. 
The Project i build to reproduce **MLOps-pipeline** with **GitHub Actions** and **Docker**

---

## Command to run the project

### Clone the project
```bash
git clone https://github.com/Satre03/virtual-diabetes-clinic-triage.git
cd virtual-diabetes-clinic-triage
```

Create and activate virtual environment\
MAC:
```bash
python3 -m venv venv
source venv/bin/activate 
```
WINDOWS:
``` bash
py -m venv venv
venv\Scripts\activate
```
Installing dependencies

```bash
pip install -r requirements.txt
```
Train model manuelly (not required)
```bash
python src/train.py
```
Build Docker-image
```bash
docker build -t ghcr.io/satre03/virtual-diabetes-clinic-triage:v0.1 .
```
Run containern
```bash
docker run -p 8000:8000 ghcr.io/satre03/virtual-diabetes-clinic-triage:v0.1
```
Open in browser:
```bash
http://localhost:8000/health
```
Excepted answer:
```bash
{"status": "ok", "model_version": "v0.1"}
```
Example payload using /predict
send with curl:
```bash
curl -Method POST http://localhost:8000/predict `
-Headers @{ "Content-Type" = "application/json" } `
-Body '{
  "age": 0.02,
  "sex": -0.044,
  "bmi": 0.06,
  "bp": -0.03,
  "s1": -0.02,
  "s2": 0.03,
  "s3": -0.02,
  "s4": 0.02,
  "s5": 0.02,
  "s6": -0.001
}'
```
Expected answer:
```bash
{"prediction": 235.9}
```

## Ta bort?
Kör containern
```bash
docker run -p 8002:8000 ghcr.io/melissawestberg/diabetes_risk_service:v0.3
```
Då nås API:t på:
```bash
http://localhost:8002/health
```






