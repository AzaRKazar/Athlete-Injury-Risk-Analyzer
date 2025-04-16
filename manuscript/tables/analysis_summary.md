# Athlete Injury Risk Analysis Summary

Analysis Date: 2025-04-02

## Dataset Overview

- Total samples: 6372
- High-risk samples: 2664 (41.8%)
- Unique athletes: 334
- Date range: 2022-01-10 to 2024-12-06

## Key Findings

### Top Risk Factors

| feature          |   cohens_d |      p_value |
|:-----------------|-----------:|-------------:|
| MaxForceSymmetry |   0.723439 | 1.47357e-141 |
| TorqueSymmetry   |   0.723439 | 1.47357e-141 |
| ForceSymmetry    |   0.700743 | 4.40281e-135 |
| ImpulseSymmetry  |   0.561588 | 2.24607e-94  |
| leftImpulse      |  -0.187761 | 1.84811e-13  |
### Athlete Clusters

- Optimal number of clusters: 2
