# Automated Insight Engine (H-001)

## 📌 Problem Statement
**Challenge Number:** H-001  
**Track:** Data Engineering & Analytics  

In the AdTech ecosystem, massive amounts of data (foot traffic logs, ad clickstreams, weather data, etc.) are generated daily.  
Currently, Account Managers manually download CSVs, join datasets, make charts, and prepare weekly PDF/Slide reports – a slow, repetitive, error-prone workflow.

### ❗ The Problem
- Manual downloading and merging of multi-source data  
- Error-prone reporting work  
- Time-consuming weekly report creation  
- No automated insights or intelligence  
- No standard formatting for client reports  

---

## 🎯 Challenge Requirements
Build a system that:
1. **Ingests multi-source data** (CSV, SQL, weather, traffic, ads).
2. **Cleans and merges data** into meaningful metrics.
3. **Generates insights using AI** (GPT‑4o or similar).
4. **Automatically creates PDF/PPTX reports**.
5. **Enables one‑click export from a UI**.

Bonus:
- Use Python (Pandas/Polars)
- Use LLMs for insights
- Output downloadable artifacts (PDF/PPT)

---

## 💡 Solution Overview

### ✔ Automated Data Pipeline
The solution ingests:
- Ad performance data (impressions, clicks, spend)
- Weather data (temperature, rainfall, condition)
- Foot traffic data (location, footfall)

### ✔ AI Insight Engine
Uses GPT‑4o (or any LLM) to:
- Evaluate campaign performance
- Identify patterns (ex: low CTR on rainy days)
- Generate executive-ready natural‑language summaries

### ✔ Report Generator
Exports:
- Beautiful PDF report (ReportLab)
- PowerPoint deck (python-pptx)

### ✔ Web UI (React)
- Upload dataset
- Trigger backend processing
- Display insights
- Download PDF/PPTX

---

## 🛠 Technology Stack

### Backend
- **FastAPI** – API layer
- **Python-Pandas** – data transformation
- **Python‑PPTX** – slide generation
- **ReportLab** – PDF export
- **OpenAI GPT‑4o** – AI insights

### Frontend
- **React.js**
- **Axios** (file upload)
- **Minimal clean UI**

---

## 🚀 Approach

### 1️⃣ Data Ingestion  
Backend accepts raw CSVs → stored in `/uploads`.

### 2️⃣ Data Processing  
- Validate required columns  
- Calculate KPIs (CTR, CPC, performance trends)  
- Merge external sources (weather, traffic)

### 3️⃣ AI Insight Generation  
Prompt LLM with computed metrics → get smart narrative insights.

### 4️⃣ Report Creation  
- Generate slide deck with charts + insights  
- Generate PDF version for executives  
- Return links to download reports

### 5️⃣ React Frontend  
Acts as a clean upload/report generation UI.

---

## 📁 Project Structure
```
backend/
  main.py
  data_processor.py
  ai_insights.py
  report_pdf.py
  report_ppt.py
  requirements.txt

frontend/
  src/
    App.jsx
    api.js
    components/
       FileUpload.jsx
  package.json

README.md
```
---

## 📝 Author
Built by **Lovkush Sharma**  
Automated for H‑001 Hackathon Challenge.

---  
