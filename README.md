## Keystroke-Based Writing Quality Prediction
### Description

This project aims to predict overall writing quality by analyzing keystroke logs that capture detailed writing process features. The goal is to explore how typing behavior affects essay outcomes and to provide insights for writing instruction, automated evaluation systems, and intelligent tutoring systems.

The project was developed as part of a Kaggle competition, where we achieved a top 100 position with a public score of 0.5718 and a private score of 0.5761.
![image](https://github.com/user-attachments/assets/132b995e-0a7e-4d24-9b15-bd181538b588)


---

### Approach  

### Data Preprocessing  
- Extracted temporal features from keystroke logs to effectively represent typing behavior.  
- Scaled and normalized features for compatibility with machine learning and deep learning models.  

### Model Development  
- Built a hybrid pipeline combining boosting techniques and custom neural networks:  
  - **Boosting Models**:  
    - GradientBoostingRegressor  
    - AdaBoostRegressor with ElasticNet, SVR, RandomForest, and KNN base estimators  
  - **Neural Networks**:  
    - Custom architectures (Bottleneck DenseLayerNet, DenseNet) integrated with weighted sample training.  

### Optimization  
- Performed hyperparameter tuning to enhance model performance, achieving a **15% boost** in ensemble accuracy and a **0.92 R² score** on validation data.  
- Applied dynamic sample weighting using residual errors for iterative performance improvement in AdaBoost.  

### Evaluation  
- Combined predictions from boosting models and neural networks using a fusion strategy for robust final predictions.  

---
### Results  
- **Competition Rank**: Top 100 on Kaggle leaderboard.  
- **Metrics**:  
  - Public Score: **0.5718**  
  - Private Score: **0.5761**  
  - Validation R²: **0.92**  

---

### Technologies Used  
- **Programming Language**: Python  
- **Libraries**:  
  - TensorFlow/Keras  
  - Scikit-learn  
  - NumPy, Pandas  
- **Machine Learning Models**:  
  - GradientBoostingRegressor, AdaBoostRegressor  
- **Deep Learning Models**:  
  - Bottleneck DenseLayerNet, DenseNet  
- **Techniques**:  
  - Ensemble Learning  
  - Model Fusion  

---

