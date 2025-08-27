# PASTA Effort Estimator (Streamlit)

Interactive Streamlit app to experiment with a PASTA-based effort/scalability model:

\[
\hat{E} = \frac{(A \times T \times V \times C \times R)^S}{M \times Au}
\]

- **A**: Assets, **T**: ThreatVectors, **V**: Vulnerabilities, **C**: Complexity, **R**: ChangeRate  
- **M**: OrgMaturity, **Au**: Automation, **S**: Scaling factor

## ✨ What’s new
- **Noise controls**: None / Gaussian / Laplace, scaled as % of mean prediction.
- **Log mode**: Compute plots and R² in **log10 space** (robust to large scales).
- **Sensitivity analysis**: One-Factor-At-a-Time (OFAT) sweep + **elasticities**:
  - Closed-form elasticities from the model.
  - Empirical elasticities via finite differences.
- **Reproducibility**: Independent seeds for data & noise.
- **Downloads**: Dataset (CSV), experiment config (JSON), and all plots (PNG).

## 🚀 Run locally
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Community Cloud
1. Push this folder to a **public GitHub repo**.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, pick your repo/branch.
4. Set **Main file path** to `app.py` and deploy.

## 🧪 Using the app
1. **Dataset Settings**: choose sample size, scaling factor **S**, and random seed.
2. **Attribute Ranges**: set ranges for A, T, V, C, R, M, Au.
3. **Noise Settings**: pick distribution and scale (% of mean predicted effort).
4. **Analysis Settings**:
   - Toggle **Log Mode** to analyze in log10 space.
   - Run **Sensitivity** by selecting a variable and number of sweep points.
5. **Validation**:
   - App reports **R² (raw)** and **R² (log10)**.
   - View **Actual vs Predicted** and **Residuals** plots.
6. **Downloads**:
   - **CSV** dataset, **JSON** config for reproducibility, and **PNG** plots.

> ⚠️ **Note**: When `ActualEffort = PredictedEffort + small noise`, R² will be optimistic. Treat this as a **sanity-check** setup. For research/IEEE results, validate on real or realistically simulated data.

## 📂 File structure
```
.
├── app.py               # Streamlit app
├── requirements.txt     # Python deps
├── runtime.txt          # Python version (Streamlit Cloud)
├── README.md            # This file
└── .gitignore
```

## 📄 Citation / Acknowledgment
If you use this tool in academic work, please cite your own paper and mention this repository as the experimental interface for model exploration and sensitivity analysis.
