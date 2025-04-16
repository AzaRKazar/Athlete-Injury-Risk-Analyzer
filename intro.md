# Introduction

Injury risk prediction in sports has traditionally relied on static threshold models, particularly in assessing inter-limb asymmetry metrics such as force, impulse, and torque. However, increasing evidence suggests these static thresholds lack context sensitivity and fail to adapt to athlete-specific variability. This limitation becomes particularly significant when considering the natural variations in athlete performance and the dynamic nature of sports training and competition.

Inter-limb asymmetry, defined as measurable differences between left and right limb performance metrics, has been widely cited as a key indicator of injury risk. Most prior research has utilized fixed cutoffs (typically 10-15%) to flag abnormal asymmetry, but this approach often overlooks natural athlete variability and task-specific demands. Furthermore, the majority of such studies have been conducted in European or Australian contexts, leaving a significant gap in understanding U.S.-based athlete populations.

This study addresses both methodological and geographic gaps by developing a machine learning-based dynamic threshold model using performance symmetry data from collegiate athletes in the U.S. We propose a novel framework that continuously adjusts risk thresholds based on each athlete's baseline variability, analyzing its effectiveness in predicting injury risk over time. Our approach represents a paradigm shift from traditional static threshold models toward personalized injury monitoring in sports science.

The growing body of research exploring the relationship between asymmetry and injury risk has demonstrated the potential of machine learning approaches. Previous studies have applied decision trees, boosting algorithms, and support vector machines to assess risk factors, yet these implementations often relied on static thresholds and overlooked the evolution of asymmetry over time. Systematic reviews, such as those by Claudino et al. (2019), emphasize the potential of artificial intelligence in sports injury modeling while noting limited application in real-time or dynamic contexts.

Our research contributes to this field by:
1. Developing a dynamic buffer thresholding framework that adapts to individual athlete variability
2. Implementing machine learning models to predict injury risk based on symmetry deviations
3. Evaluating model performance across multiple quarters to track individual athlete risk progression
4. Introducing athlete-specific alerting thresholds
5. Addressing the geographic gap in the literature by focusing on U.S. collegiate athletes

This study aims to provide a more robust and personalized approach to injury risk prediction, moving beyond traditional static thresholds to consider the complex interplay of individual athlete characteristics, temporal patterns, and performance metrics. By incorporating machine learning techniques with dynamic thresholding, we seek to improve early-warning capabilities and ultimately reduce preventable injuries in collegiate athletics. 