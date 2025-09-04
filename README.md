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
- 🔎 Key Insights
  - Dataset had missing values → imputed successfully.
  - Dataset is imbalanced → fewer potable samples compared to non-potable.
  - Some features (pH, Hardness, Sulfate) show clear impact on water Potability.
  - Outliers exist in some columns (e.g., Solids, Hardness) which may affect model performance.
  - Cleaned dataset is now ready for Week 3: Model Building.

## 🚀 How to Run
1. Clone this repository or download the files.
2. Open Google Colab or Jupyter Notebook.
3. Upload and run the notebooks from the notebooks/ folder.
4. Install dependencies using:
   pip install -r requirements.txt
