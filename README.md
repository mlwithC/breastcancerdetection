#  Breast Cancer Detection using Machine Learning

This is a beginner ML project where I explore how to predict if a tumor is benign or malignant using Logistic Regression and other models.

>  Status: In Progress

---

##  Dataset
- Source: `sklearn.datasets.load_breast_cancer`

---

##  Goals
-  Clean & analyze the data
-  Build ML model using Logistic Regression
-  Evaluate performance with accuracy & classification report
-  Visualize metrics like ROC-AUC, confusion matrix
-  Deploy as a simple web app (Streamlit)

---

##  Tools Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## Work Done So Far

-  Loaded the Breast Cancer dataset using `load_breast_cancer()`
-  Converted dataset into Pandas DataFrame
-  Explored the dataset: structure, missing values, class balance
-  Handled missing values using

Model Evaluation
 Accuracy: 0.9736842105263158

 Classification Report:

               precision    recall  f1-score   support

           0       0.98      0.95      0.96        43
           1       0.97      0.99      0.98        71

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114

Compared the exiting model with The Decision Tree and The Random forest


---

## Model Comparison

| Model               | Accuracy     |
|--------------------|--------------|
| Logistic Regression| 0.9561403509 |
| Decision Tree      | 0.9473684210 |
| Random Forest      | 0.9649122807 |

 **Random Forest performed the best** with the highest accuracy.


Model Accuracy Comparison (Breast Cancer Detection)

| Model                     | Accuracy   |
| ------------------------- | ---------- |
| Decision Tree             | 94.73%     |
| Random Forest             | 96.49%     |
| K-Nearest Neighbors (KNN) | 94.73%     |
| Support Vector Machine    | 97.36%     |

 SVM currently has the highest accuracy among all tested models.

*Trained and evaluated 4 ML models

*Computed accuracy, confusion matrix, and classification report for each

*Compared performance and updated results 




