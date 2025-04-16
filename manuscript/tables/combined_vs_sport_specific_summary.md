# Combined vs. Sport-Specific Model Comparison

Analysis Date: 2025-04-02

## Overview

This analysis compared the performance of models trained on all sports combined versus models trained on individual sports. The comparison helps determine whether sport-specific asymmetry patterns are distinct enough to warrant separate models, or whether a single model trained on all sports performs as well or better.

## Key Findings

1. **Combined Model Performance**: The best model for the combined dataset was RF_No_Balancing with an F1 score of 1.000.
2. **Sport-Specific Performance**: The average F1 score across best sport-specific models was 0.991 ± 0.021.
3. **Statistical Significance**: The difference between combined and sport-specific models was not statistically significant (p=0.6698).
4. **Top Performing Sports**: The sports with highest individual model performance were combined (F1=1.000), football (F1=1.000), womes lax (F1=1.000).

## Model Type Analysis

### Combined Dataset Model Performance

| Model | F1 Score | Accuracy | ROC AUC |
|-------|----------|----------|---------|
| RF_No_Balancing | 1.000 | 1.000 | 1.000 |
| RF_with_SMOTE | 1.000 | 1.000 | 1.000 |
| RF_with_SMOTEENN | 0.998 | 0.998 | 1.000 |
| RF_with_Class_Weights | 1.000 | 1.000 | 1.000 |

### Sports Where Combined Model Outperformed Sport-Specific Model

| Sport | Sport F1 | Combined F1 | Improvement |
|-------|----------|-------------|-------------|
| womens volleyball | 0.941 | 1.000 | 6.3% |
| softball | 0.941 | 1.000 | 6.3% |

### Sports Where Sport-Specific Model Outperformed Combined Model

| Sport | Sport F1 | Combined F1 | Difference |
|-------|----------|-------------|------------|
| combined | 1.000 | 1.000 | -0.0% |
| football | 1.000 | 1.000 | -0.0% |
| womes lax | 1.000 | 1.000 | -0.0% |
| mens soccer | 1.000 | 1.000 | -0.0% |
| womens soccer | 1.000 | 1.000 | -0.0% |
| womens basketball | 1.000 | 1.000 | -0.0% |
| mens lax | 1.000 | 1.000 | -0.0% |
| baseball | 1.000 | 1.000 | -0.0% |
| mens basketball | 1.000 | 1.000 | -0.0% |
| track and field | 1.000 | 1.000 | -0.0% |
| tennis | 1.000 | 1.000 | -0.0% |

## Implications

1. **Both Approaches Viable**: There was no significant difference between combined and sport-specific models, suggesting that both approaches can be effective depending on the implementation context.
2. **Hybrid Approach Potential**: Some sports benefited from the combined model while others performed better with sport-specific models. This suggests a potential hybrid approach where certain sports use dedicated models while others leverage the combined model.
3. **Data Requirements**: The combined model benefits from larger sample sizes, which may explain its strong performance. Sport-specific models require sufficient data for each sport to be reliable.
4. **Transfer Learning Opportunity**: The success of the combined model suggests potential for transfer learning, where models trained on data-rich sports can be adapted for use with sports having limited data availability.
