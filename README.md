# decision-trees-with-python
Learn how to build a decision tree classifier that determines which drug might be appropriate for a future patient with the same illness. 

## Project Overview
This project uses a Decision Tree Classifier to predict which drug a patient is likely to respond to based on demographic and clinical features such as age, sex, blood pressure, and cholesterol levels. The model is trained on a small Kaggle dataset (https://www.kaggle.com/datasets/pablomgomez21/drugs-a-b-c-x-y-for-decision-trees) and evaluated using standard classification metrics. The goal is to demonstrate how decision trees can be applied to interpretable, rule-based medical decision support systems.

## Dataset Description
**Features**
1. Age: Patient age (integer)
2. Sex: Male or Female (categorical)
3. Blood Pressure: Low, Normal, High (categorical)
4. Cholesterol: Normal or High (categorical)

**Target**
1. Drug type prescribed to the patient

## Methodology
Categorical features were encoded using label encoding. The dataset was split into training and testing sets using an 80/20 ratio. A Decision Tree Classifier was selected due to its interpretability and ability to model non-linear relationships without feature scaling.

## Model Evaluation
The model achieved perfect scores (1.00) for precision, recall, and F1-score across all drug classes. This means the Decision Tree Classifier perfectly predicted the drug type for all 40 test instances. Take note that while this is excellent for this specific dataset, in real-world scenarios, it’s common to see less-than-perfect scores, and a perfect score can sometimes indicate potential overfitting or a very easily separable dataset.

## Warning
This model is for educational purposes only and should not be used for real-world medical decision-making.
