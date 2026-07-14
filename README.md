# 🌾 OptiCrop — Smart Agricultural Production Optimization Engine

An AI-powered crop recommendation system that leverages Machine Learning to help farmers make data-driven decisions. Input your soil and climate parameters, and OptiCrop predicts the most suitable crop for maximum yield and sustainability.

<!-- ![OptiCrop Banner](https://img.shields.io/badge/OptiCrop-Smart%20Farming-27ae60?style=for-the-badge&logo=seedling) -->

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 🎯 Project Objective

OptiCrop aims to develop an advanced software system that utilizes data-driven insights to optimize agricultural production for different crops. By integrating key environmental factors, it provides intelligent recommendations to farmers for maximizing yields and resource efficiency.

---

## ✨ Features

- 🤖 **ML-Powered Predictions** — Trained model with **95%+ accuracy** across 22+ crop types
- 📊 **7 Input Parameters** — Nitrogen, Phosphorous, Potassium, Temperature, Humidity, pH, Rainfall
- 🌱 **22+ Crop Types** — Rice, Wheat, Maize, Cotton, Coffee, Mango, and more
- 🎨 **Modern UI** — Clean, responsive interface built with Bootstrap 5
- ⚡ **Real-time Results** — Instant crop recommendations with seasonal growing info

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3, Bootstrap 5, Poppins Font |
| **Backend** | Python 3.10+, Flask 3.1 |
| **ML/Data** | scikit-learn, Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Gunicorn, Render |

---

## 📁 Project Structure

```
OptiCrop/
├── .gitignore                # Git ignore configuration
├── Demo Video/               # Project demonstration video
├── README.md                 # Project documentation
├── render.yaml               # Render deployment blueprint
└── Project Files/            # Root folder for all project files
    ├── app.py                    # Flask application (routes & prediction logic)
    ├── model.pkl                 # Trained ML model
    ├── scaler.pkl                # Feature scaler
    ├── requirements.txt          # Python dependencies
    ├── gunicorn_config.py        # Production server config
    ├── Crop_recommendation.csv   # Training dataset
    ├── opticrop_analysis.py      # Data analysis & model training script
    ├── generate_dataset.py       # Dataset generation utility
    ├── templates/
    │   ├── index.html            # Home page
    │   ├── about.html            # About page
    │   └── findyourcrop.html     # Crop prediction form & results
    ├── static/
    │   └── css/
    │       └── style.css         # Custom styles
    └── plots/                    # EDA & model evaluation visualizations
        ├── crop_distribution.png
        ├── confusion_matrix.png
        ├── kmeans_clusters.png
        └── ...
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AP24110010258/OptiCrop.git
   cd OptiCrop
   ```

2. **Navigate to the Project Files directory**
   ```bash
   cd "Project Files"
   ```

3. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://127.0.0.1:5000
   ```

---

## 📊 Input Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| Nitrogen (N) | Soil nitrogen content ratio | kg/ha |
| Phosphorous (P) | Soil phosphorous content ratio | kg/ha |
| Potassium (K) | Soil potassium content ratio | kg/ha |
| Temperature | Average temperature | °C |
| Humidity | Relative humidity | % |
| pH | Soil pH value | 0–14 |
| Rainfall | Annual rainfall | mm |

---

## 🌾 Supported Crops

| Category | Crops |
|----------|-------|
| **Cereals** | Rice, Maize |
| **Pulses** | Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil |
| **Fruits** | Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut |
| **Cash Crops** | Cotton, Jute, Coffee |

---

## 🔬 ML Pipeline

```
Raw Data → Data Cleaning → EDA & Visualization → Feature Scaling → Model Training → Evaluation → Deployment
```

**Algorithms explored:**
- K-Nearest Neighbors (KNN)
- Logistic Regression
- K-Means Clustering
- Decision Trees

**Model Performance:** 95%+ accuracy on the test set.

---

## 🌍 Impact

### Environmental
- Optimizes water and fertilizer usage
- Reduces environmental damage from poor crop selection
- Promotes sustainable farming practices

### Business
- Improves agricultural productivity and profitability
- Reduces financial losses from unsuitable crop selection
- Supports data-driven agricultural planning

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for smarter, sustainable farming<br>
  <strong>OptiCrop</strong> — Smart Agricultural Production Optimization Engine
</p>
