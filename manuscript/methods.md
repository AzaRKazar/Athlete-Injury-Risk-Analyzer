# Methods

## Data Collection and Preprocessing

The study utilized force plate data collected from athletes using VALD ForceDecks equipment in a Division I athletic conference. The raw data consisted of bilateral force measurements including average force, impulse, maximum force, and torque for both left and right legs. The initial dataset contained 972 measurements collected between January 2022 and July 2024. 

The data cleaning process involved several key steps. First, we performed initial data cleaning by removing calibration and repetition columns, converting timestamp data to date format, eliminating duplicate entries, and removing rows with zero values in key metrics. Second, we implemented comprehensive outlier treatment, which included identifying and handling infinite values, applying the Interquartile Range (IQR) method for outlier detection, and capping values beyond Q1 - 1.5 * IQR and Q3 + 1.5 * IQR. After preprocessing, the final dataset contained 972 measurements.

## Symmetry Metric Calculation

To assess inter-limb symmetry, we computed four key symmetry ratios using the collected force plate data. These ratios were calculated with left values as numerators to maintain consistency. The Force Symmetry metric, calculated as Left Average Force divided by Right Average Force, showed a mean of 1.052344 with a standard deviation of 0.125037 and a range from 0.740896 to 1.350772. The Impulse Symmetry metric, derived from Left Impulse divided by Right Impulse, exhibited a mean of 1.072752, a standard deviation of 0.151197, and a range from 0.709789 to 1.420651. The Max Force Symmetry metric, calculated as Left Max Force divided by Right Max Force, demonstrated a mean of 1.053861, a standard deviation of 0.125259, and a range from 0.738484 to 1.358941. Finally, the Torque Symmetry metric, derived from Left Torque divided by Right Torque, showed identical statistical properties to Max Force Symmetry, with a mean of 1.053861, standard deviation of 0.125259, and range from 0.738484 to 1.358941. Each metric was subjected to the same outlier treatment process as described above to ensure data quality.

## Risk Classification Methodology

### Asymmetry Classification Framework
Our classification system categorizes biomechanical asymmetry based on thresholds derived from empirical data. For each symmetry metric, we established specific thresholds that defined acceptable ranges for low asymmetry performance. These thresholds were determined as follows: ForceSymmetry had a low asymmetry range of 0.806 to 1.154, derived from force measurements of 250/310 to 300/260; ImpulseSymmetry had a low asymmetry range of 0.577 to 1.563, derived from impulse measurements of 150/260 to 250/160; MaxForceSymmetry had a low asymmetry range of 0.818 to 1.143, derived from maximum force measurements of 270/330 to 320/280; and TorqueSymmetry had a low asymmetry range of 0.806 to 1.154, derived from torque measurements of 250/310 to 300/260.

### Dynamic Buffer Implementation
To improve the sensitivity and specificity of our asymmetry classification, we implemented a dynamic buffer approach that adapts to the natural variability in the dataset. For each symmetry metric, we began with standard deviation analysis, calculating the standard deviation for each symmetry metric across the entire population. Next, we established a dynamic buffer using these standard deviations, with a configurable buffer factor (set to 1.0 in our implementation). Using these dynamic buffers, we assigned asymmetry categories as follows: Low Asymmetry for values within the established thresholds, Medium Asymmetry for values falling just outside thresholds (within one buffer), and High Asymmetry for values beyond the threshold plus buffer. While these categories are labeled using risk terminology in the literature, we acknowledge that our study does not directly connect these classifications to injury outcomes.

### Composite Asymmetry Score
To determine an overall asymmetry category for each athlete, we implemented a "highest asymmetry" approach. The asymmetry categories from three primary metrics (ForceSymmetry, MaxForceSymmetry, and TorqueSymmetry) were evaluated. If any metric showed "High Asymmetry," the athlete was categorized as having High Asymmetry. If no metrics showed "High Asymmetry" but at least one showed "Medium Asymmetry," the athlete was categorized as having Medium Asymmetry. Only when all metrics showed "Low Asymmetry" was the athlete categorized as having Low Asymmetry. This comprehensive approach ensures that any significant asymmetry is appropriately flagged, while also providing a clear overall assessment.

### Statistical Validation of Asymmetry Categories
To ensure that our classification system was statistically valid, we conducted a comprehensive validation procedure. We performed one-way ANOVA tests for ForceSymmetry, MaxForceSymmetry, and TorqueSymmetry to verify that the metrics significantly differed across asymmetry categories. Tukey's HSD post-hoc tests were conducted to determine if differences between each pair of asymmetry categories (Low-Medium, Medium-High, Low-High) were statistically significant. For ImpulseSymmetry, which showed potential non-normal distribution, we implemented Kruskal-Wallis tests to ensure robust statistical validation. Additionally, Cohen's d effect sizes were calculated between each pair of asymmetry categories to quantify the magnitude of differences, providing further validation of the meaningful separation between asymmetry levels.

## Cross-Sport Analysis Framework

Our study expanded beyond sport-specific analyses to include a comprehensive cross-sport evaluation framework. We implemented two complementary analytical approaches to understand symmetry patterns across different athletic populations.

First, we conducted sport-specific modeling, where individual Random Forest models were trained and evaluated for each sport separately. This approach allowed us to identify sport-specific patterns of asymmetry and assess whether predictive models perform differently across various athletic disciplines. We developed models for 12 different sports including football (n=972), women's lacrosse (n=429), men's soccer (n=321), women's soccer (n=316), women's basketball (n=242), men's lacrosse (n=215), baseball (n=205), men's basketball (n=144), women's volleyball (n=124), softball (n=94), track and field (n=87), and tennis (n=35).

Second, we implemented an all-sports combined model that leveraged the full dataset across all sports. This approach allowed us to identify universal patterns of asymmetry that transcend sport-specific movement patterns. The combined model was trained on data from all 3,184 athletes, enabling a more robust assessment of general asymmetry patterns and providing greater statistical power for identifying subtle relationships in the data.

To compare the effectiveness of these approaches, we conducted statistical testing to determine whether sport-specific models significantly outperformed the combined model. This analysis included paired comparisons of performance metrics (accuracy, F1 score, and ROC AUC) between sport-specific and combined models, as well as ANOVA testing to identify significant differences in model performance across sports.

## Model Development and Evaluation

### Model Architecture
We implemented four distinct Random Forest classification models to address class imbalance in our dataset, each employing different balancing techniques. All models were implemented using scikit-learn's RandomForestClassifier with 100 decision trees and a random state of 42 for reproducibility. The feature engineering process included three primary symmetry metrics (ForceSymmetry, MaxForceSymmetry, and TorqueSymmetry) that were selected based on their performance in preliminary analysis and their biomechanical relevance.

Model (1), the No Balancing Model, served as our baseline, trained on the original imbalanced dataset without any balancing techniques. It represents the natural distribution of asymmetry categories in our dataset, with a higher prevalence of Low Asymmetry cases compared to Medium and High Asymmetry cases. Model (2), the SMOTE Model, utilized Synthetic Minority Over-sampling Technique to balance the dataset by generating synthetic samples for minority classes. SMOTE creates new instances of minority classes by interpolating between existing minority class samples, helping to address the class imbalance while preserving the underlying data distribution.

Model (3), the SMOTEENN Model, combined SMOTE with Edited Nearest Neighbors, which not only oversamples minority classes but also removes samples that are misclassified by their neighbors. This hybrid approach helps to create a more balanced dataset while also cleaning the decision boundaries between classes. The SMOTEENN model was ultimately selected as our primary model due to its ability to handle both oversampling and undersampling in a single step. Model (4), the Class Weights Model, addressed class imbalance through class weights, assigning higher weights to minority classes during model training. The weights were inversely proportional to class frequencies, with weights of 1, 2, and 3 assigned to Low, Medium, and High Asymmetry classes respectively. This approach modifies the model's learning process to pay more attention to minority class samples without changing the original data distribution.

### Addressing Methodological Challenges

#### Data Leakage Mitigation
During our analysis, we identified a potential data leakage issue where the target variable was directly derived from features used in the model. To address this challenge, we implemented a rigorous methodological validation process that included three key components.

First, we developed models without direct target-related features by removing MaxForceSymmetry and TorqueSymmetry (which were used in threshold calculations to define the target variable) from the feature set. These models relied solely on raw measurements (leftAvgForce, leftImpulse, leftMaxForce, leftTorque, rightAvgForce, rightImpulse, rightMaxForce, rightTorque) and indirect symmetry metrics (ForceSymmetry, ImpulseSymmetry) that were not directly used in the target definition.

Second, we created alternative target definitions that were statistically independent from the features used in modeling. This approach ensured that the model was not simply learning the rule used to create the target variable but was instead identifying genuine patterns in the data that have predictive value.

Third, we implemented athlete-based cross-validation to prevent data leakage across multiple tests from the same athlete. By ensuring that all tests from the same athlete were either entirely in the training set or entirely in the test set, we eliminated the possibility of the model learning athlete-specific patterns rather than generalizable asymmetry indicators.

#### Performance Decline Prediction
To create a more clinically relevant predictive framework, we developed a longitudinal analysis approach that focused on predicting performance decline over time rather than simply describing current asymmetry states. This approach involved several key steps.

First, we identified athletes with multiple tests in our database and sorted their data chronologically. For each athlete with at least two tests, we calculated changes in key symmetry metrics (ForceSymmetry, MaxForceSymmetry, ImpulseSymmetry, TorqueSymmetry) between consecutive tests.

Second, we created multiple target variables to capture different aspects of performance decline:
- "Any Symmetry Worse" - a binary indicator of any worsening in symmetry metrics
- "Significant Symmetry Worse" - a binary indicator of significant worsening (>10% change) in any symmetry metric
- "Number of Metrics Worse" - a count of how many symmetry metrics worsened between tests
- "Decline Severity" - a categorical variable classifying decline as None, Minimal, Moderate, or Severe

Third, we trained predictive models using only data available at the time of the initial test to predict future performance decline. This approach creates a genuinely predictive model that could be used in clinical settings to identify athletes at risk for developing worsening asymmetry before it occurs.

### Model Training and Testing Protocol
For all four models, we implemented a rigorous training and testing protocol. The dataset was split into 80% training and 20% testing sets using stratified sampling to maintain the same class distribution in both sets. Three primary symmetry metrics (ForceSymmetry, MaxForceSymmetry, and TorqueSymmetry) were selected as predictive features based on their biomechanical relevance and preliminary analysis. Each model was trained on its respective processed dataset (original, SMOTE-resampled, SMOTEENN-resampled, or class-weighted). All trained models were saved using joblib for reproducibility and future deployment. Finally, models were validated on a separate unseen dataset to assess generalization performance, with predictions mapped back to asymmetry categories for interpretation.

### Cross-Validation Framework
To ensure robust model evaluation and prevent overfitting, we implemented a comprehensive cross-validation framework. We employed stratified 5-fold cross-validation to maintain class distribution across all folds, providing a more reliable assessment of model performance. For each fold, we calculated multiple performance metrics including accuracy, precision (macro), recall (macro), and F1-score (macro) to provide a holistic evaluation of model performance. Standard deviations across folds were calculated for each metric to assess model stability and reliability. Cross-validation results were visualized across all models to facilitate direct comparison and identify the most robust approach.

### Feature Importance Analysis
To address our second research question regarding the discriminative power of different symmetry metrics, we conducted a thorough feature importance analysis. We calculated permutation feature importance for each model by randomly shuffling each feature and measuring the resulting decrease in model performance, providing a robust measure of feature relevance. We also extracted feature importance directly from the Random Forest models based on mean decrease in impurity (MDI) to determine each feature's contribution to the classification decision. Feature importance was compared across all four models to identify consistent patterns and ensure robust conclusions about feature relevance. Comprehensive visualizations were created to illustrate relative feature importance across models, helping to identify the most critical symmetry metrics for asymmetry classification.

### ROC Analysis and Model Comparison
We implemented a comprehensive model comparison framework using Receiver Operating Characteristic (ROC) analysis. ROC curves and Area Under the Curve (AUC) scores were calculated for each model, with micro-average ROC curves providing an overall assessment of model discriminative ability. We analyzed prediction consistency across all four models on unseen data, identifying the degree of agreement between models and potential cases requiring further investigation. Model performance was assessed across different temporal splits of the data to evaluate performance stability over time and identify potential drift in prediction patterns.

### Evaluation Framework
For our primary question regarding the comparison between dynamic and static threshold models, we conducted direct model comparisons using paired t-tests and analyzed classification consistency across different time periods. This approach allowed us to assess the effectiveness of our dynamic thresholding approach compared to traditional static methods.

For our first secondary question concerning classification accuracy, we implemented stratified k-fold cross-validation (k=5) to assess model stability and prevent data leakage. We evaluated model performance using a comprehensive set of metrics including accuracy, precision, recall, and F1-score. Additionally, we conducted temporal analysis of classification accuracy across quarterly periods and generated detailed confusion matrices to understand error patterns and model behavior.

To address our second secondary question about feature importance, we employed multiple analytical approaches. We conducted permutation importance analysis for each symmetry metric to determine their individual discriminative power. We also utilized SHAP (SHapley Additive exPlanations) values to assess feature interactions and understand how different metrics influence the model's predictions. Furthermore, we performed correlation analysis between different symmetry metrics and examined the temporal stability of feature importance to ensure robust feature selection.

### Statistical Analysis
We conducted comprehensive statistical analysis to validate our findings through both descriptive and inferential approaches. Our descriptive statistical analysis encompassed three main areas. First, we performed distribution analysis, examining symmetry metric distributions, analyzing temporal patterns in measurements, and assessing asymmetry category distribution across the dataset.

Second, we conducted statistical validation using ANOVA tests for ForceSymmetry, MaxForceSymmetry, and TorqueSymmetry, along with a Kruskal-Wallis test for ImpulseSymmetry. We performed significance testing across asymmetry categories and analyzed standard deviations for each symmetry metric.

Third, we implemented comprehensive model performance evaluation, which included comparing four different Random Forest models, analyzing classification accuracy across asymmetry categories, evaluating model stability and consistency, and assessing feature importance and interactions. This multi-faceted approach ensured robust validation of our findings and provided comprehensive insights into the effectiveness of our asymmetry classification system.

## Implementation Details

The analysis was implemented using Python, leveraging key libraries for different aspects of the work. We used pandas (version 1.5.3) for data manipulation and analysis, numpy (version 1.24.3) for numerical computations, scikit-learn (version 1.2.2) for machine learning models including the RandomForestClassifier, imbalanced-learn (version 0.10.1) for implementing SMOTE and SMOTEENN resampling techniques, matplotlib (version 3.7.1) and seaborn (version 0.12.2) for visualization, and scipy (version 1.10.1) for statistical testing. 

For model persistence and reproducibility, we utilized joblib (version 1.2.0) to save and load trained models. The codebase was organized into modular components, including data preprocessing scripts, symmetry calculation and risk classification modules, model training and evaluation pipelines, visualization utilities, and statistical analysis tools. All code was version-controlled and documented to facilitate reproducibility of our findings.

This modular structure facilitated efficient development, testing, and maintenance of the analysis pipeline, while ensuring that our methodology could be transparently evaluated and replicated by other researchers.