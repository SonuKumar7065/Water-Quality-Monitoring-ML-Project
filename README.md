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

## ✅ Week 2 Task – Data Cleaning & Exploratory Data Analysis (EDA)
- Handled missing values in ph, Sulfate, and Trihalomethanes using mean imputation.
- Removed duplicate rows.
- Performed EDA with visualizations:
  - Histograms for feature distributions
  - Boxplots for outlier detection
  - Correlation heatmap
  - Count plot for potable vs non-potable water
  - Comparative boxplots (e.g., pH vs Potability, Hardness vs Potability)
- Saved cleaned dataset as cleaned_water_potability.csv
- Saved notebook as Week2_Water_Quality.ipynb.
- 🔎 Key Insights-
  - Dataset had missing values → imputed successfully.
  - Dataset is imbalanced → fewer potable samples compared to non-potable.
  - Some features (pH, Hardness, Sulfate) show clear impact on water Potability.
  - Outliers exist in some columns (e.g., Solids, Hardness) which may affect model performance.
  - Cleaned dataset is now ready for Week 3: Model Building.

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
