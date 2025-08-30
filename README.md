🌍 Water Quality Monitoring using Machine Learning

This project is part of my AI/ML Internship (Week 1–X). The aim is to monitor and predict water potability using ML techniques. The dataset contains various chemical properties of water, and the target variable indicates whether water is safe to drink (1) or not safe (0).

📌 Week 1 – Data Understanding & Exploration
✅ Tasks Completed:

Imported required Python libraries.

Loaded dataset (water_potability.csv).

Explored dataset using:

.info() – Data types & memory usage

.describe() – Statistical summary

.isnull().sum() – Missing value check

Handled missing values (replaced with mean).

Visualized dataset with:

Correlation Heatmap

Potability Distribution (0/1)

Saved cleaned dataset (cleaned_water_quality.csv) for further use.

📊 Dataset Information

Rows (before cleaning): 3276

Columns: 10

Target Variable: Potability (0 = Not Safe, 1 = Safe)

🖼️ Sample Outputs
Correlation Heatmap

(Place screenshot here after running Notebook)

Water Potability Distribution

(Place screenshot here after running Notebook)

📂 Repository Structure
📁 Water_Quality_ML_Project
 ┣ 📄 Week1_Water_Quality.ipynb   # Week 1 Notebook
 ┣ 📄 water_potability.csv        # Raw dataset
 ┣ 📄 cleaned_water_quality.csv   # Cleaned dataset
 ┗ 📄 README.md                   # Project Documentation

🚀 Next Steps (Week 2)

Perform Exploratory Data Analysis (EDA) in detail (boxplots, histograms, feature distribution).

Train baseline ML models (Logistic Regression, Random Forest).

Evaluate with accuracy, precision, recall, F1-score.

👨‍💻 Author

Sonu Kumar
AI/ML Internship Project – 2025
