# XAU/USD Trading ML Project

A machine learning project for XAU/USD (Gold) trading analysis and prediction using Random Forest algorithms with MetaTrader 5 integration.

## 📊 Project Overview

This project implements a machine learning approach to analyze and predict XAU/USD price movements using historical data and Random Forest algorithms. The system integrates with MetaTrader 5 for live trading capabilities and includes comprehensive data analysis through Jupyter notebooks.

## 🚀 Features

- **Machine Learning Models**: Random Forest implementation for price prediction
- **MetaTrader 5 Integration**: Direct connection to MT5 for live data and trading
- **ONNX Model Support**: Optimized model deployment with ONNX format
- **Historical Data Analysis**: Comprehensive analysis of XAU/USD H1 data from 2018-2024
- **Jupyter Notebook**: Interactive analysis and model development environment
- **Secure Credentials Management**: Template-based credential setup

## 📁 Project Structure

```
ml/
├── xauusd.ipynb                 # Main Jupyter notebook for analysis
├── mt5_login.py                 # MetaTrader 5 connection utilities
├── setup_credentials_template.py # Credentials setup template
├── xauusd_best_model.onnx       # Trained ONNX model
├── random_forest.mq5            # MetaTrader 5 Expert Advisor source
├── random_forest.ex5            # Compiled MetaTrader 5 Expert Advisor
├── XAUUSDm_H1_201801020600_202412310000.csv # Historical data
├── credentials/                 # API keys and login credentials
├── results/                     # Model results and analysis outputs
└── data/                        # Additional data files
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- MetaTrader 5 terminal
- Jupyter Notebook/Lab
- Required Python packages (see requirements below)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd ml
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

*Note: Create requirements.txt with the following packages:*
```
pandas
numpy
scikit-learn
MetaTrader5
jupyter
onnx
onnxruntime
matplotlib
seaborn
```

### Step 4: Setup Credentials

1. Copy `setup_credentials_template.py` to `setup_credentials.py`
2. Fill in your MetaTrader 5 credentials:
   ```python
   MT5_LOGIN = "your_login"
   MT5_PASSWORD = "your_password"
   MT5_SERVER = "your_server"
   ```
3. Run the setup script:
   ```bash
   python setup_credentials.py
   ```

## 📈 Usage

### 1. Data Analysis

Open the main Jupyter notebook:
```bash
jupyter notebook xauusd.ipynb
```

The notebook includes:
- Historical data loading and preprocessing
- Exploratory data analysis
- Feature engineering
- Model training and validation
- Performance evaluation

### 2. MetaTrader 5 Integration

Use the MT5 login utility to connect:
```python
from mt5_login import connect_mt5

# Connect to MT5
if connect_mt5():
    print("Connected to MetaTrader 5")
    # Your trading logic here
```

### 3. Model Deployment

The trained ONNX model can be used for predictions:
```python
import onnxruntime as ort

# Load the model
session = ort.InferenceSession('xauusd_best_model.onnx')

# Make predictions
predictions = session.run(None, {'input': your_data})
```

### 4. MetaTrader Expert Advisor

1. Copy `random_forest.ex5` to your MetaTrader 5 `Experts` folder
2. Attach the EA to an XAU/USD chart
3. Configure the EA parameters as needed

## 📊 Model Performance

The Random Forest model is trained on:
- **Dataset**: XAU/USD H1 data (January 2018 - December 2024)
- **Features**: OHLC prices, technical indicators, time-based features
- **Target**: Price direction prediction

*Results and performance metrics can be found in the `results/` directory after running the analysis.*

## 🔧 Configuration

### MetaTrader 5 Settings

Ensure your MT5 terminal has:
- Algorithmic trading enabled
- Required symbols (XAUUSD) in Market Watch
- Sufficient account balance for testing

### Model Parameters

Key parameters can be adjusted in the notebook:
- `n_estimators`: Number of trees in the forest
- `max_depth`: Maximum depth of trees
- `min_samples_split`: Minimum samples required to split a node

## 🤖 Trading Strategy

The implemented strategy includes:
- **Entry Signals**: Based on Random Forest predictions
- **Risk Management**: Position sizing and stop-loss mechanisms
- **Time Filters**: Trading session restrictions
- **Market Conditions**: Volatility and trend filters

## ⚠️ Risk Disclaimer

**This project is for educational and research purposes only.** 

- Past performance does not guarantee future results
- Trading involves substantial risk of loss
- Never trade with money you cannot afford to lose
- Always test strategies thoroughly on demo accounts first
- Consider seeking advice from qualified financial professionals

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue in the repository.

## 🙏 Acknowledgments

- MetaTrader 5 for providing the trading platform API
- Scikit-learn for machine learning algorithms
- ONNX for model optimization and deployment

---

**Happy Trading! 📈** 