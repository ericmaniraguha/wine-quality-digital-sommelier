# 🍷 Wine Quality Prediction — Digital Sommelier

> **From Raw Wine Chemistry to Machine Learning and an Interactive Digital Sommelier**
[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter\&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas\&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy\&logoColor=white)](https://numpy.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Digital%20Sommelier-FF4B4B?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![Joblib](https://img.shields.io/badge/Joblib-Model%20Serialization-green)](https://joblib.readthedocs.io/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github\&logoColor=white)](https://github.com/ericmaniraguha/wine-predictions)


An end-to-end **machine learning application** that predicts wine quality from physicochemical laboratory measurements and presents the prediction through an interactive **Streamlit Digital Sommelier**.

The project uses the **UCI Wine Quality Dataset**, containing red and white Vinho Verde wines from Portugal.



---

## 🎯 Project Goal

The goal of this project is to answer a practical data science question:

> **Can the sensory quality of wine be predicted from its physicochemical laboratory measurements?**

The project takes raw wine chemistry data through a complete machine learning workflow:

```text
Raw Wine Data
     ↓
Data Understanding
     ↓
Data Quality & Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Feature Scaling
     ↓
Unsupervised Learning
     ↓
Supervised Learning
     ↓
Model Comparison
     ↓
Hyperparameter Tuning
     ↓
Model Evaluation
     ↓
Model Interpretation
     ↓
Model Serialization
     ↓
Streamlit Digital Sommelier
```

---

# 🍇 Dataset

The project uses the **UCI Wine Quality Dataset** based on Vinho Verde wines from Portugal.

Two datasets are included:

- `winequality-red.csv`
- `winequality-white.csv`

Each wine is described using physicochemical measurements together with a sensory **quality score**.

### Features

| Feature | Description |
|---|---|
| `fixed acidity` | Concentration of non-volatile acids |
| `volatile acidity` | Volatile acidity level |
| `citric acid` | Citric acid concentration |
| `residual sugar` | Sugar remaining after fermentation |
| `chlorides` | Chloride concentration |
| `free sulfur dioxide` | Free SO₂ concentration |
| `total sulfur dioxide` | Total SO₂ concentration |
| `density` | Density of the wine |
| `pH` | Acidity level |
| `sulphates` | Sulphate concentration |
| `alcohol` | Alcohol percentage |
| `quality` | Sensory wine quality score |

---

# 🔬 Project Workflow

## 01 — Project Overview & Problem Framing

The project begins by defining wine quality prediction as a supervised machine learning problem.

The target variable is:

```text
quality
```

The physicochemical measurements are used as predictive features.

---

## 02 — Data Loading & Understanding

The red and white wine datasets are loaded and examined to understand:

- Dataset dimensions
- Feature types
- Target distribution
- Descriptive statistics
- Red vs. white wine differences
- Potential data-quality issues

---

## 03 — Data Quality & Cleaning

The raw datasets are assessed and prepared for analysis and modelling.

The cleaning workflow considers:

- Missing values
- Duplicate observations
- Data types
- Outliers
- Invalid or unusual values
- Feature distributions
- Data consistency

The objective is to ensure that the modelling dataset is reliable and reproducible.

---

## 04 — Exploratory Data Analysis

Exploratory analysis is used to understand the structure of the data and relationships between wine chemistry and quality.

The analysis includes:

- Feature distributions
- Wine quality distribution
- Correlation analysis
- Outlier analysis
- Red vs. white wine comparison
- Feature-to-quality relationships
- Target correlation analysis

Generated visualizations are stored in:

```text
figures/
```

Examples include:

- Correlation heatmap
- Feature distributions
- Quality distribution
- Red vs. white comparison
- Outlier boxplots
- Feature relationships with quality

---

# 🧮 Feature Engineering & Preprocessing

The project prepares the physicochemical features for machine learning through preprocessing and feature engineering.

This includes:

- Selecting predictive features
- Separating features and target
- Train/test splitting
- Feature scaling
- Preparing data for different algorithms

Feature-related logic is also maintained in:

```text
wine_features.py
```

---

# 🔍 Unsupervised Learning

Unsupervised learning is used to explore the underlying structure of the wine dataset.

The project investigates clustering and dimensionality reduction techniques, including:

### Clustering

- K-Means
- Cluster selection
- Dendrogram analysis

### Dimensionality Reduction

- Principal Component Analysis (PCA)
- Explained variance
- Two-dimensional PCA projection

The objective is to understand whether wines naturally form groups based on their physicochemical characteristics.

Relevant visualizations are available in:

```text
figures/10_*.png
```

---

# 🤖 Supervised Learning

Multiple supervised learning approaches are evaluated to identify a strong model for predicting wine quality.

The modelling workflow includes:

- Baseline models
- Cross-validation
- Model comparison
- Hyperparameter tuning
- Test-set evaluation

The project evaluates model performance using metrics such as:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination

---

# ⚙️ Model Evaluation

Models are compared using both cross-validation and independent test-set evaluation.

Generated comparison results include:

```text
figures/11_model_comparison_cv.png
figures/13_model_comparison_test.png
```

Residual diagnostics are also generated to examine prediction errors:

```text
figures/13_residual_diagnostics.png
```

This provides a more complete assessment of model performance rather than relying on a single metric.

---

# 🧠 Model Interpretation

Predictive performance alone is not enough. The project also investigates **why the model makes its predictions**.

Several interpretation techniques are included.

### Feature Importance

```text
figures/14_feature_importance.png
```

### Permutation Importance

```text
figures/14_permutation_importance.png
```

### Partial Dependence

```text
figures/14_partial_dependence.png
```

These analyses help identify which physicochemical properties contribute most to predicted wine quality and how changes in individual features affect model predictions.

---

# 💾 Model Artifacts

The final trained model is saved for use by the application.

```text
models/
├── wine_quality_model.joblib
├── feature_ranges.json
└── model_card.json
```

### `wine_quality_model.joblib`

Serialized machine learning model used for prediction.

### `feature_ranges.json`

Stores the expected feature ranges used by the application to validate user inputs.

### `model_card.json`

Documents important information about the trained model, including its intended use and relevant modelling information.

Keeping these artifacts separately from the notebook makes it possible to use the trained model directly in the application without retraining it every time.

---

# 🍷 Digital Sommelier

The trained model is packaged into an interactive **Streamlit** web application.

The application is implemented in:

```text
app.py
```

The user can provide physicochemical measurements and receive a predicted wine quality score.

### Application workflow

```text
User Input
    ↓
Input Validation
    ↓
Feature Preparation
    ↓
Trained Model
    ↓
Quality Prediction
    ↓
Digital Sommelier Result
```

This converts the analytical notebook into a practical user-facing machine learning application.

---

# 🖥️ Running the Application

## 1. Clone the Repository

```bash
git clone https://github.com/ericmaniraguha/wine-predictions.git
cd wine-predictions
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Launch the Digital Sommelier

```bash
streamlit run app.py
```

The Streamlit application will then be available in your browser.

---

# 📓 Explore the Analysis

The complete analytical workflow is documented in:

```text
wine_quality_analysis.ipynb
```

The notebook is organized into the following sections:

| # | Section |
|---:|---|
| 01 | Project Overview & Problem Framing |
| 02 | Import Libraries |
| 03 | Load the Dataset |
| 04 | Understand the Data Structure |
| 05 | Data Quality Assessment |
| 06 | Exploratory Data Analysis |
| 07 | Feature Engineering & Target |
| 08 | Train/Test Split |
| 09 | Feature Scaling |
| 10 | Unsupervised Learning |
| 11 | Supervised Learning |
| 12 | Hyperparameter Tuning |
| 13 | Model Evaluation & Comparison |
| 14 | Model Interpretation |
| 15 | Business / Winery Insights |
| 16 | Final Model |
| 17 | Save Model |
| 18 | Streamlit Digital Sommelier |
| 19 | Conclusion & Limitations |

Each major section includes a **🔎 Highlights** explanation covering:

- **What we do**
- **Why it matters**
- **What to look for in the output**

---

# 📁 Repository Structure

```text
wine-predictions/
│
├── app.py
├── wine_features.py
├── wine_quality_analysis.ipynb
├── requirements.txt
├── README.md
│
├── winequality-red.csv
├── winequality-white.csv
│
├── figures/
│   ├── 05_outliers_boxplots.png
│   ├── 06_correlation_heatmap.png
│   ├── 06_feature_distributions.png
│   ├── 06_quality_distribution.png
│   ├── 06_red_vs_white.png
│   ├── 06_target_correlation.png
│   ├── 06_top_features_vs_quality.png
│   ├── 09_scaling_effect.png
│   ├── 10_choosing_k.png
│   ├── 10_dendrogram.png
│   ├── 10_pca_projection.png
│   ├── 10_pca_variance.png
│   ├── 11_knn_k_curve.png
│   ├── 11_linear_coefficients.png
│   ├── 11_model_comparison_cv.png
│   ├── 13_model_comparison_test.png
│   ├── 13_residual_diagnostics.png
│   ├── 14_feature_importance.png
│   ├── 14_partial_dependence.png
│   └── 14_permutation_importance.png
│
├── models/
│   ├── feature_ranges.json
│   ├── model_card.json
│   └── wine_quality_model.joblib
│
└── outputs/
```

---

# 🛠️ Technologies

### Programming & Analysis

- Python
- Jupyter Notebook
- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- K-Means
- PCA
- Regression / predictive modelling
- Cross-validation
- Hyperparameter tuning

### Model Explainability

- Feature Importance
- Permutation Importance
- Partial Dependence Analysis

### Application

- Streamlit
- Joblib

### Development

- Git
- GitHub

---

# 💡 Business & Winery Insights

The project demonstrates how wine laboratory measurements can be transformed into data-driven insights.

Potential applications include:

- Wine quality screening
- Quality-control support
- Laboratory data analysis
- Production monitoring
- Wine classification
- Data-driven experimentation
- Decision-support systems

However, the model should be considered a **decision-support tool**, not a replacement for professional sensory evaluation.

---

# ⚠️ Limitations

### Dataset limitations

The dataset represents Vinho Verde wines from Portugal and may not generalize to all wine varieties, regions, or production methods.

### Subjective quality scores

Wine quality is based on sensory evaluation and can therefore contain subjectivity and variation between assessors.

### Limited features

The dataset primarily contains physicochemical measurements. Other factors that may influence wine quality include:

- Grape variety
- Vineyard conditions
- Soil
- Climate
- Fermentation process
- Aging
- Storage
- Winemaking techniques
- Human sensory perception

These factors are not fully represented.

### Prediction is not professional tasting

The Digital Sommelier provides a machine learning prediction based on available chemistry data. It should not be interpreted as a professional sommelier's sensory assessment.

---

# 🔮 Future Improvements

Possible future development includes:

- Deploying the application to the cloud
- Dockerizing the application
- Adding a REST API for model inference
- Adding MLflow experiment tracking
- Implementing model monitoring
- Automated model retraining
- Adding additional wine datasets
- Adding wine-style classification
- Adding prediction confidence/uncertainty
- Building a wine recommendation engine
- Integrating natural-language explanations
- Adding an LLM-powered conversational Digital Sommelier

---

# 📚 Dataset Reference

**Wine Quality Dataset**

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009).

*Modeling wine preferences by data mining from physicochemical properties.*

Decision Support Systems, 47(4), 547–553.

Dataset: **UCI Machine Learning Repository — Wine Quality Dataset**

---

# 👨‍💻 Author

**Eric Maniraguha**

Data Engineer | Data Scientist | ICT Systems & Digital Transformation Consultant

📧 **Email:** ericmaniraguha@gmail.com

🔗 **GitHub:** `github.com/ericmaniraguha`

---

# ⭐ Project Summary

**Wine Quality Prediction — Digital Sommelier** demonstrates a complete journey from raw laboratory data to an operational machine learning application:

```text
🍇 Raw Wine Chemistry
        ↓
🧹 Data Cleaning
        ↓
📊 Exploratory Analysis
        ↓
🔧 Feature Engineering
        ↓
🤖 Machine Learning
        ↓
📈 Model Evaluation
        ↓
🧠 Model Interpretation
        ↓
💾 Model Serialization
        ↓
🍷 Digital Sommelier
```

> **Turning wine chemistry into data-driven quality predictions. 🍷**