# Athlete Injury Risk Analyzer

This project focuses on analyzing athlete performance data using symmetry metrics and building predictive models to classify athletes into different risk categories (Low Risk, Medium Risk, High Risk).

---

## **Roadmap**

### **1. Data Preparation**
1. **Data Collection:**
   - Source athlete performance data with metrics such as `leftAvgForce`, `rightAvgForce`, `ImpulseSymmetry`, etc.
   - Ensure data contains necessary identifiers (`sbuid`, `testDateUtc`).

2. **Pivoting the Data:**
   - Reshape the dataset using a pivot table to organize metrics as columns for each athlete and test date.

3. **Preprocessing:**
   - Calculate symmetry metrics (`ForceSymmetry`, `ImpulseSymmetry`, `MaxForceSymmetry`, `TorqueSymmetry`).
   - Handle missing values, remove duplicates, and cap outliers using the IQR method.

4. **Threshold Definition:**
   - Define thresholds for symmetry metrics based on domain knowledge or data distribution.
   - Apply dynamic buffer logic for flexible risk categorization.

---

### **2. Hypothesis Testing**
1. Group symmetry metrics by risk categories (`Low Risk`, `Medium Risk`, `High Risk`).
2. Perform statistical tests (e.g., ANOVA or Kruskal-Wallis) to assess if differences in metrics across categories are significant.
3. Document results and determine which metrics are most impactful for risk categorization.

---

### **3. Model Development**
1. **Define Features and Target:**
   - Features: Symmetry metrics (`ForceSymmetry`, `MaxForceSymmetry`, `TorqueSymmetry`).
   - Target: `RiskCategory` (encoded as Low = 0, Medium = 1, High = 2).

2. **Handle Class Imbalance:**
   - Experiment with different techniques:
     - SMOTE Oversampling
     - SMOTEENN (Combined Oversampling and Undersampling)
     - Class Weights in Random Forest

3. **Train Models:**
   - Build Random Forest models for each technique.
   - Compare results of the following:
     - Model 1: SMOTE Oversampling
     - Model 2: No Balancing
     - Model 3: SMOTEENN
     - Model 4: Class Weights

4. **Evaluate Models:**
   - Metrics: Accuracy, F1-Score, Precision, Recall, Confusion Matrix.
   - Visualize confusion matrices and accuracy comparisons for each model.

---

### **4. Prediction on Unseen Data**
1. Preprocess new datasets to calculate symmetry metrics.
2. Use trained models to predict risk categories for new athletes.
3. Compare predictions across all models for consistency and reliability.

---

### **5. Visualizations and Insights**
1. Plot risk distribution across metrics.
2. Visualize confusion matrices for all models.
3. Display quarterly trends for selected athletes (if temporal data is available).
4. Summarize hypothesis testing results and model comparisons in charts.

---

### **6. Deployment and Documentation**
1. Deploy the final models and preprocessing pipeline using a tool like Streamlit.
2. Document the project with:
   - A clear README file summarizing the project.
   - Model insights and key findings.
3. Provide options for future enhancements:
   - Expand to include additional symmetry metrics.
   - Test on a broader dataset with different sports.

---

