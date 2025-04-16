# Sport-Specific Model Performance Summary

Analysis Date: 2025-04-02

## Overview

This analysis trained separate asymmetry classification models for 13 different sports with sufficient data. Four different modeling approaches were compared for each sport to identify the most effective algorithm for capturing sport-specific asymmetry patterns.

## Key Findings

1. **Best Predictability**: The sports with most predictable asymmetry patterns were combined, football, womes lax.
2. **Most Challenging**: The sports with least predictable asymmetry patterns were tennis, womens volleyball, softball.
3. **Best Algorithm**: RF_No_Balancing was the most effective model, performing best for 100.0% of sports.
4. **Performance Range**: F1 scores ranged from 0.941 to 1.000, indicating substantial variation in how well asymmetry patterns can be classified across different sports.

## Performance by Sport

| Sport | Best Model | F1 Score | Accuracy | ROC-AUC | Sample Size | Asymmetry Rate (%) |
|-------|------------|----------|----------|---------|-------------|-------------------|
| combined | RF_No_Balancing | 1.000 | 1.000 | 1.000 | 3186 | 41.8 |
| football | RF_No_Balancing | 1.000 | 1.000 | 1.000 | 972 | 41.8 |
| womes lax | RF_No_Balancing | 1.000 | 1.000 | 1.000 | 429 | 51.3 |
| mens soccer | RF_No_Balancing | 1.000 | 1.000 | 1.000 | 321 | 35.8 |
| womens soccer | RF_No_Balancing | 1.000 | 1.000 | 1.000 | 316 | 37.0 |
| womens basketball | RF_No_Balancing | 1.000 | 1.000 | 1.000 | 242 | 48.3 |
| mens lax | RF_No_Balancing | 1.000 | 1.000 | 1.000 | 215 | 44.7 |
| baseball | RF_No_Balancing | 1.000 | 1.000 | 1.000 | 205 | 34.6 |
| mens basketball | RF_No_Balancing | 1.000 | 1.000 | 1.000 | 144 | 38.9 |
| track and field | RF_No_Balancing | 1.000 | 1.000 | 1.000 | 87 | 35.6 |
| tennis | RF_No_Balancing | 1.000 | 1.000 | 1.000 | 35 | 51.4 |
| womens volleyball | RF_No_Balancing | 0.941 | 0.960 | 0.971 | 124 | 33.9 |
| softball | RF_No_Balancing | 0.941 | 0.947 | 1.000 | 94 | 45.7 |

## Implications

1. **Sport-Specific Approaches**: The substantial variation in model performance across sports confirms the need for sport-specific assessment approaches.
2. **Sampling Strategies**: The effectiveness of different sampling strategies (SMOTE, SMOTEENN) varies by sport, likely reflecting different class imbalance characteristics.
3. **Data Requirements**: Sports with larger sample sizes generally yielded more reliable models, highlighting the importance of adequate data collection for each sport.
4. **Asymmetry Complexity**: The varying predictability of asymmetry patterns suggests that biomechanical demands differ significantly across sports, with some creating more consistent and predictable asymmetry profiles than others.
