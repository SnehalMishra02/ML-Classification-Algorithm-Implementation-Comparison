# Machine Learning Classification Algorithms: Implementation and Comparison

## Overview

This project implements and compares different classification algorithms using the **Iris dataset**. The goal is to evaluate and analyze the performance of various classifiers based on accuracy and other metrics.

## Algorithms Implemented

- Logistic Regression
- k-Nearest Neighbors (k-NN)
- Decision Trees
- Random Forests
- Support Vector Machines (SVM)

## Dataset

The **Iris dataset** consists of 150 samples with 4 features each, classified into three species. The dataset is split into **70% training** and **30% testing** to ensure fair evaluation.

## Preprocessing Steps

- **Standardization**: Applied to improve model performance, particularly for SVM and k-NN.
- **Stratified Split**: Ensures class balance in training and testing sets.

## Model Performance Comparison

| Algorithm           | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 91%      |
| k-NN                | 91%      |
| Decision Tree       | 98%      |
| Random Forest       | 89%      |
| SVM                 | 93%      |

## Key Observations

- **Decision Trees** achieved the highest accuracy (98%) but might be **overfitting**.
- **Random Forests** performed slightly worse than Decision Trees due to reduced overfitting measures.
- **SVM and Logistic Regression** provided balanced results, indicating they generalize well.
- **k-NN's** accuracy depends on the value of **k**; hyperparameter tuning can further optimize results.

## How to Run the Code

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/ml-classification-comparison.git
   cd ml-classification-comparison
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```
3. Run the script:
   ```bash
   python classification_comparison.py
   ```

## Future Enhancements

- **Hyperparameter tuning** using Grid Search
- **Try alternative datasets** for more robust results
- **Implement additional classifiers** (e.g., Neural Networks, XGBoost)

---

**Author:** Snehal Mishra\
**License:** MIT\
**Contributions:** Feel free to submit issues or pull requests!

