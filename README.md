# Commodity AI: Petrol Price Forecasting & Multi-Agent ML Studio ⛽🤖

An end-to-end Machine Learning forecasting platform and autonomous Multi-Agent Studio built with Python and Streamlit. The application predicts retail petrol prices across **10 major Indian metropolitan cities** for the next 7 days using historical fuel trends, live web scraping, Brent crude oil dynamics, and the USD/INR foreign exchange rate.

> 📖 **For complete technical details, mathematical formulations, and internal execution logic, see [`PROJECT_EXPLANATION.md`](file:///d:/Projects/ml%20project/PROJECT_EXPLANATION.md).**

---

## 🏛️ Project Architecture

```
d:\Projects\ml project\
├── .gitignore                         # Git exclusion rules (virtualenvs, binaries, caches)
├── README.md                          # Repository overview & quickstart guide
├── PROJECT_EXPLANATION.md             # In-depth architectural & algorithmic manual
├── requirements.txt                   # Core Python dependencies
├── app.py                             # Unified Executive Streamlit Terminal
│
├── data/
│   └── ahmedabad_petrol_daily.csv     # 1,216 daily observations (2021–2024)
├── petrol_module/                     # Dedicated Time-Series Fuel Forecasting Module
│   ├── data_loader.py                 # Chronological data ingestion & validation
│   ├── feature_engineering.py         # Stationary delta creation, rolling stats, lag generators
│   ├── models.py                      # Naive Baseline, Linear, Ridge, Random Forest, XGBoost
│   ├── evaluator.py                   # Out-of-sample chronological evaluation (MAE, RMSE, R²)
│   ├── forecast.py                    # 7-day iterative roll-forward engine with macro transmission
│   └── live_fetcher.py                # Real-time multi-city web scraper & OMC fallback registry
├── agents/                            # Multi-Agent Cooperative ML System
│   ├── data_agent.py                  # Ingestion, inspection, cleaning, standard scaling
│   ├── model_agent.py                 # Model dispatching, hyperparameter tuning, model training
│   ├── eval_agent.py                  # Classification and regression metrics calculation
│   └── orchestrator.py                # Pipeline orchestrator, timers, audit trail
└── saved_models/                      # Serialized ML model artifacts (.joblib)
    └── .gitkeep                       # Preserves folder in version control
```

---

## ⚡ Core Highlights & Capabilities

### 1. Multi-City Live Fuel Rate Tracking
- Scrapes live retail pump prices across **10 Indian cities & states**: Ahmedabad, Surat, Vadodara, Rajkot, Mumbai, Delhi, Bengaluru, Hyderabad, Chennai, and Kolkata.
- Includes state-specific VAT and cess taxation models with fallback to verified Oil Marketing Company (OMC) daily benchmarks.

### 2. Methodological Rigor (Stationary Delta Formulation)
- **Extrapolation Fix**: Tree-based models (Random Forest, XGBoost) fail to extrapolate trends on raw price levels ($P_t$). Our pipeline models the stationary daily price change ($\Delta P_t = P_t - P_{t-1}$), enabling reliable forward forecasting.
- **Strict Anti-Leakage**: 80% train / 20% test sequential split with zero future lookahead.
- **Naive Baseline Benchmark**: Compares all machine learning models against a persistent benchmark ($P_{t+1} = P_t$).

### 3. Bi-Directional 7-Day Forecasting
- Generates realistic price forecasts supporting both **downward price cuts (▼)**, **flat pauses (▬)**, and **upward hikes (▲)**.
- Models international macroeconomic transmission ($\approx 0.072~\text{INR/L}$ per $\$1$ Brent Crude move) across 4 selectable market scenarios:
  - 📉 *Bearish Crude / Price Reduction Cycle*
  - ⚖️ *Dynamic Market Volatility*
  - 🎯 *OMC Rate Pause (Regulated Stability)*
  - 📈 *Bullish Crude Squeeze*
  - Interactive Crude Oil Shock Slider ($-10\%$ to $+10\%$).

### 4. Autonomous Multi-Agent Machine Learning Studio
- General-purpose machine learning studio where specialized agents collaborate:
  - **DataAgent**: Cleans, imputes missing values, scales features via `StandardScaler`, and splits data.
  - **ModelAgent**: Trains Regression models (Random Forest, Gradient Boosting, Ridge, Decision Tree) or Classification models (Random Forest, Gradient Boosting, Logistic Regression, Decision Tree).
  - **EvalAgent**: Generates MAE, RMSE, $R^2$ or Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
  - **WorkflowOrchestrator**: Coordinates execution, logs audit events, and saves artifacts to disk.

---

## 🚀 How to Run the Application

1. Open PowerShell in the project directory:
   ```powershell
   .\venv\Scripts\streamlit run app.py
   ```
2. Open your browser to `http://localhost:8501`.
3. Switch between views using the top navigation bar or sidebar:
   - **⛽ Petrol Price Predictor (Multi-City)**: Live price tracking, macroeconomic scenarios, and 7-day projected trajectory.
   - **📊 Model Performance Benchmarks**: Chronological test fit curves, feature importances, and real-time live accuracy verification.
   - **🤖 Multi-Agent Machine Learning Studio**: Train classification or regression models on custom datasets with real-time agent audit logs.
