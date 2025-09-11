# Water Quality Monitoring - AI/ML Internship Project

This repository contains my work for the Edunet AI/ML Internship Project on **Water Quality Monitoring**.

## 📂 Repository Structure
- `dataset/` → contains water_potability.csv dataset  
- `notebooks/` → contains Colab notebooks  
- `README.md` → project documentation  
- `requirements.txt` → Python dependencies  

## ✅ Week 1 Task – Data Loading & Initial Exploration
- Imported required Python libraries.
- Imported dataset `water_potability.csv`
- Performed data exploration (info, head, describe, missing values, shape)
- Checked dataset shape (rows × columns)
- Performed basic exploration:
  - .info() → dataset structure
  - .describe() → statistical summary
  - .isnull().sum() → checked missing values
- Saved notebook as Week1_Water_Quality.ipynb

## ✅ Week 2: Data Preprocessing & Cleaning  

During Week 2, the focus was on **cleaning and preparing the dataset** for machine learning.  

### 🔹 Key Steps
- Handled **missing values** in critical features:  
  - `ph` → filled using mean imputation  
  - `Sulfate` → filled using mean imputation  
  - `Trihalomethanes` → filled using mean imputation  
- Removed **duplicate records** to ensure dataset quality  
- Converted dataset into a **clean and analysis-ready version**  

### 🔹 Exploratory Data Analysis (EDA)
- Generated **summary statistics** (`.describe()`)  
- Created **distribution plots** (histograms, boxplots)  
  ![Distribution of Features](images/distribution.png)  

- Used a **heatmap correlation matrix** to identify relationships between features  
  ![Correlation Heatmap](images/heatmap.png)  

### 🔹 Output
- Saved cleaned dataset as:  
  - 📂 `dataset/cleaned_water_potability.csv`  

### 🔹 Insights
- Significant missing values were present initially, especially in `ph`, `Sulfate`, and `Trihalomethanes`.  
- Correlation analysis showed moderate relationships between some features.  
- Dataset imbalance was observed (`Potability = 0` has more samples than `Potability = 1`).  


## ✅ Week 3: Machine Learning Models
- Built classification models to predict **water potability**  

### 📊 Models Implemented
1. **Logistic Regression** → Baseline linear model  
2. **Random Forest Classifier** → Non-linear, handles imbalance with `class_weight="balanced"`  

### 🔹 Results
| Model                | Accuracy | ROC-AUC |
|-----------------------|----------|---------|
| Logistic Regression   | ~0.63    | ~0.66   |
| Random Forest         | ~0.72    | ~0.79   |

👉 Random Forest outperforms Logistic Regression in both accuracy and AUC.  

---

## 📊 Model Results & Visualizations  

### Confusion Matrix – Random Forest  
![Confusion Matrix RF](images/confusion_matrix_rf.png)  

### ROC Curve Comparison  
![ROC Curve](images/roc_curve_rf.png)  

### Feature Importance  
![Feature Importance](images/feature_importance.png)  

### Top 5 Features  
![Top 5 Features](images/top5_features.png)  

---

### 🔹 Key Insights
- **Random Forest** is more reliable for this dataset.  
- **Important features**: `pH`, `Solids`, `Sulfate`, and `Trihalomethanes`.  
- Dataset imbalance still affects performance; future improvements can use **SMOTE** or **Ensemble Methods**.  

## 🚀 How to Run
1. Clone this repository or download the files.
2. Open Google Colab or Jupyter Notebook.
3. Upload and run the notebooks from the notebooks/ folder.
4. Install dependencies using:
   pip install -r requirements.txt
