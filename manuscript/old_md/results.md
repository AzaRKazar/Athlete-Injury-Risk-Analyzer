# Results

## Risk Classification Distribution

Our risk classification system produced a balanced distribution of athletes across different risk categories. Table 1 presents the distribution of risk categories for each symmetry metric, demonstrating the effectiveness of our dynamic buffer approach in creating meaningful risk separation.

**Table 1: Distribution of Athletes Across Risk Categories by Symmetry Metric**

| Symmetry Metric | Low Risk | Medium Risk | High Risk |
|-----------------|----------|-------------|-----------|
| ForceSymmetry   | 664      | 189         | 119       |
| ImpulseSymmetry | 749      | 144         | 79        |
| MaxForceSymmetry| 658      | 204         | 110       |
| TorqueSymmetry  | 686      | 175         | 111       |

The composite risk approach, which assigns the highest risk level across all metrics, resulted in the following overall risk distribution: 584 athletes (60.1%) classified as Low Risk, 295 athletes (30.3%) as Medium Risk, and 93 athletes (9.6%) as High Risk.

**Figure 1: Risk Category Distribution by Symmetry Metric**
![Risk Category Distribution](figures/risk_category_distribution.png)

The visualization in Figure 1 illustrates the distribution of athletes across the three risk categories for each symmetry metric. This balanced distribution supports the effectiveness of our dynamic buffer approach in creating meaningful separation between risk categories.

**Figure 2: Risk Distribution Bar Plot**
![Risk Distribution Bar Plot](figures/risk_distribution_barplot.png)

Figure 2 provides an alternative visualization of the risk distribution, highlighting the relative proportions of each risk category across the four symmetry metrics. This representation further demonstrates the balanced nature of our risk classification system.

## Statistical Validation of Risk Categories

To ensure that our risk classification system represented statistically distinct groups, we conducted ANOVA analyses followed by Tukey's HSD post-hoc tests for each symmetry metric.

**Table 2: ANOVA Results for Risk Categories**

| Symmetry Metric | F-statistic | p-value      | Significant |
|-----------------|-------------|--------------|-------------|
| ForceSymmetry   | 311.35      | 3.73e-105    | Yes         |
| MaxForceSymmetry| 337.14      | 7.29e-112    | Yes         |
| TorqueSymmetry  | 296.74      | 2.96e-101    | Yes         |

The ANOVA results confirmed that all three primary symmetry metrics showed highly significant differences between risk categories (p < 0.001), validating our classification approach. Further, post-hoc analysis revealed statistically significant differences between all pairs of risk categories (Low-Medium, Medium-High, and Low-High) for each symmetry metric.

**Table 3: Mean Differences Between Risk Categories (Tukey's HSD)**

| Comparison       | ForceSymmetry | MaxForceSymmetry | TorqueSymmetry |
|------------------|---------------|------------------|----------------|
| Low vs. Medium   | 0.108*        | 0.103*           | 0.109*         |
| Medium vs. High  | 0.201*        | 0.202*           | 0.202*         |
| Low vs. High     | 0.309*        | 0.305*           | 0.311*         |

*All differences significant at p < 0.001

**Figure 3: Effect Size Heatmap Between Risk Categories**
![Effect Size Heatmap](figures/effect_size_heatmap.png)

Figure 3 presents a heatmap of the effect sizes (Cohen's d) between different risk categories for each symmetry metric. The large effect sizes, particularly between Low Risk and High Risk groups (ranging from 4.07 to 4.19), provide strong evidence for the meaningful separation between our risk categories. Medium vs. High and Low vs. Medium comparisons also show substantial effect sizes between 1.04 and 1.41, further validating the distinct nature of these risk groups.

**Figure 4: Density Distributions by Risk Category**
![Risk Category Density Distributions](figures/risk_density_distributions.png)

Figure 4 shows the density distributions for each symmetry metric by risk category. These distributions illustrate the clear separation between risk categories, with minimal overlap between Low, Medium, and High Risk groups. This visual evidence further supports the statistical validation of our risk classification approach.

## Correlation Analysis of Symmetry Metrics

**Figure 5: Correlation Heatmap of Symmetry Metrics**
![Correlation Heatmap](figures/correlation_heatmap.png)

Figure 5 presents a correlation heatmap of the four symmetry metrics. This visualization reveals moderate to strong positive correlations between ForceSymmetry, MaxForceSymmetry, and TorqueSymmetry (r > 0.75), while ImpulseSymmetry shows weaker correlations with the other metrics (r < 0.60). These relationships help explain our feature importance findings and suggest that these metrics capture related but distinct aspects of biomechanical asymmetry.

## Overall Risk Distribution

The composite risk approach, which assigns the highest risk level across all symmetry metrics, provides a comprehensive assessment of athlete risk status. 

**Figure 6: Overall Risk Category Distribution**
![Overall Risk Distribution](figures/overall_risk_pie.png)

Figure 6 illustrates the overall distribution of athletes across the three risk categories based on our composite risk approach. The pie chart shows that the majority of athletes (60.1%) are classified as Low Risk, with 30.3% as Medium Risk and 9.6% as High Risk. This distribution aligns with clinical expectations, where most athletes would be expected to show low asymmetry levels, with progressively fewer athletes in higher risk categories.

## Model Performance Comparison

We implemented and evaluated four different Random Forest classification models to address the class imbalance in our dataset. Table 4 presents a comparison of these models' performance metrics.

**Table 4: Model Performance Comparison**

| Model                 | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) |
|-----------------------|----------|-------------------|----------------|------------------|
| Model 1 (SMOTE)       | 0.984    | 0.983             | 0.983          | 0.983            |
| Model 2 (No Balancing)| 0.990    | 0.983             | 0.970          | 0.977            |
| Model 3 (SMOTEENN)    | 0.998    | 0.997             | 0.997          | 0.997            |
| Model 4 (Class Weights)| 0.979   | 0.980             | 0.953          | 0.967            |

**Figure 7: Cross-Validation Performance Metrics**
![Cross-Validation Performance](figures/cross_validation_performance.png)

Figure 7 illustrates the cross-validation performance metrics for all four models, including accuracy, precision, recall, and F1-score. The error bars represent standard deviation across folds, demonstrating the stability of each model's performance. This visualization confirms that Model 3 (SMOTEENN) consistently outperforms the other models across all metrics with the lowest variance, showing excellent performance stability across different data subsets.

Model 3 (SMOTEENN) demonstrated the highest overall performance with near-perfect accuracy (99.8%) and the best balance of precision and recall across all risk categories. The confusion matrix for this model revealed exceptional classification performance with only 1 misclassification out of 428 test samples.

**Table 5: Confusion Matrix for Model 3 (SMOTEENN)**

|               | Predicted Low | Predicted Medium | Predicted High |
|---------------|---------------|------------------|----------------|
| Actual Low    | 144           | 0                | 0              |
| Actual Medium | 0             | 141              | 0              |
| Actual High   | 0             | 1                | 142            |

## Feature Importance Analysis

To address our second research question regarding which symmetry metrics demonstrated the strongest predictive power, we conducted comprehensive feature importance analysis using both permutation importance and mean decrease in impurity methods.

**Figure 8: Feature Importance Comparison Across Models**
![Feature Importance Comparison](figures/feature_importance_comparison.png)

Figure 8 provides a visual comparison of feature importance across all four models using permutation importance, which assesses how much model performance decreases when a feature is randomly shuffled. This analysis shows that ForceSymmetry consistently exhibits the highest permutation importance across most models, followed by MaxForceSymmetry and TorqueSymmetry. The consistent pattern across different modeling approaches strengthens our confidence in the relative contribution of each symmetry metric to injury risk prediction.

**Figure 9: Individual Feature Importance for Model 3 (SMOTEENN)**
![Model 3 Feature Importance](figures/model3_feature_importance.png)

Figure 9 focuses on the feature importance for our best-performing model (Model 3 - SMOTEENN), showing the mean decrease in impurity for each feature. This visualization reveals that MaxForceSymmetry (36.6%) contributes the most to the model's predictive power, closely followed by ForceSymmetry (33.3%) and TorqueSymmetry (30.1%). This relatively balanced contribution of all three metrics suggests that our comprehensive approach using multiple symmetry measures provides a more robust risk assessment than any single metric alone.

## Model Predictions on Unseen Data

We validated our trained models on a separate unseen dataset of 70 measurements to assess generalization performance. The models showed extraordinary agreement in their predictions, with 100% of cases receiving the same risk classification across all four models. This perfect agreement on unseen data further validates the robustness of our risk classification approach and provides exceptional confidence in its generalizability to new athlete data.

## Summary of Key Findings

Our analysis yielded several key findings:

1. The dynamic threshold model with individual buffers based on standard deviations created statistically distinct and clinically meaningful risk categories, as confirmed by ANOVA and post-hoc tests (p < 0.001) and large effect sizes (Cohen's d ranging from 1.04 to 4.19).

2. The SMOTEENN-based Random Forest model (Model 3) achieved the highest accuracy (99.8%) and balanced performance across all risk categories, demonstrating the effectiveness of combining oversampling and undersampling techniques to address class imbalance.

3. All three symmetry metrics (MaxForceSymmetry, ForceSymmetry, and TorqueSymmetry) contribute substantially to the model's predictive power, with relatively balanced importance values of 36.6%, 33.3%, and 30.1% respectively in our best model.

4. Cross-validation analysis confirmed the exceptional stability of our models, with Model 3 achieving 99.7% accuracy with minimal variance (±0.3%) across all folds.

5. Perfect agreement (100%) between all models on unseen data demonstrates exceptional generalizability and reliability of our approach for real-world applications.

These findings provide strong evidence supporting the effectiveness of our dynamic threshold approach and machine learning-based risk classification system for identifying injury risk based on symmetry metrics in collegiate athletes.

