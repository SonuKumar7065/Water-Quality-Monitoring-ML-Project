# Water Quality Monitoring - AI/ML Internship Project

This repository contains my work for the Edunet AI/ML Internship Project on **Water Quality Monitoring**.

## 📂 Repository Structure
- `dataset/` → contains water_potability.csv dataset  
- `notebooks/` → contains Colab notebooks  
- `README.md` → project documentation  
- `requirements.txt` → Python dependencies  

## ✅ Week 1 Task
- Imported required Python libraries.
- Imported dataset `water_potability.csv`
- Performed data exploration (info, head, describe, missing values, shape)
- Checked dataset shape (rows × columns)
- Saved cleaned dataset for next week’s tasks

## ✅ Week 2 Task – Data Cleaning & Exploratory Data Analysis (EDA)
- Handled missing values in ph, Sulfate, and Trihalomethanes using mean imputation.
- Removed duplicate rows to ensure dataset integrity.
- Performed exploratory data analysis (EDA) with visualizations:
- Histograms for feature distributions
- Boxplots for outlier detection
- Correlation heatmap to check feature relationships
- Count plot for Potable vs Non-Potable distribution
- Comparative boxplots (e.g., pH vs Potability, Hardness vs Potability)
- Saved cleaned dataset as:
dataset/cleaned_water_potability.csv
🔎 Key Insights
Dataset had missing values → imputed successfully.
Dataset is imbalanced → fewer potable samples compared to non-potable.
Some features (pH, Hardness, Sulfate) show clear impact on water Potability.
Outliers exist in some columns (e.g., Solids, Hardness) which may affect model performance.
Cleaned dataset is now ready for Week 3: Model Building.

## 🚀 How to Run
1. Open Google Colab
2. Upload the notebook from `notebooks/Week1_Water_Quality.ipynb`
3. Run all cells
