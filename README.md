SMS Spam Classification using Machine Learning and NLP
By: Dhiraj Jethani 
Anmol Shrivastava 

Abstract
As the volume of Short Message Service (SMS) communication continues to rise, so does the prevalence of unwanted spam messages, necessitating effective spam classification systems. This research project focuses on developing and implementing a robust SMS spam classification model using Natural Language Processing (NLP) techniques and machine learning algorithms. The project applies methods like feature engineering, data cleaning, and model evaluation to create an effective spam classifier, achieving high accuracy with the Random Forest algorithm.

Table of Contents
Introduction
Literature Review
Methodology
Results
Conclusion
Future Work
References
1. Introduction
The proliferation of SMS has led to a significant increase in spam messages, posing security and privacy threats to users. This project develops an SMS spam classification model using machine learning and Natural Language Processing (NLP). The goal is to create an effective system that can accurately classify messages as spam or ham to improve user experience and reduce unwanted content.

1.1 Background
Mobile communication has become ubiquitous, but with it comes the challenge of distinguishing legitimate messages from spam. Traditional approaches have been insufficient due to the dynamic nature of spam tactics. This project aims to address these challenges using machine learning and NLP.

1.2 Motivation
The need for effective spam detection mechanisms has become critical as SMS continues to be a popular communication tool. This research is motivated by the desire to develop an automated solution to improve user experience and safeguard against malicious content.

1.3 Objectives
Dataset Exploration and Analysis
Feature Engineering
Data Cleaning
Model Building
Model Evaluation
Real-world Predictions
1.4 Scope of the Research
The research focuses primarily on SMS spam classification and aims to provide an efficient solution to the problem. The methods used can be generalized for other text classification tasks.

2. Literature Review
This section outlines the various approaches taken in prior research for SMS spam detection, from traditional rule-based systems to modern machine learning and NLP techniques. Key topics covered include rule-based systems, NLP techniques, machine learning algorithms, and ensemble methods.

3. Methodology
3.1 Dataset
The "SMS Spam Collection" dataset, obtained from Kaggle, is used for training and evaluating the model. It contains labeled SMS messages as either spam or ham.

3.2 Exploratory Data Analysis (EDA)
EDA is performed to understand the dataset's characteristics, including its distribution and imbalance. Key visualizations and statistical methods are used for this purpose.

3.3 Feature Engineering
Handling Imbalanced Dataset: Oversampling was used to balance the dataset.
Creating New Features: Features like word count, currency symbols, and numbers in the message are added to improve model performance.
3.4 Data Cleaning
Several text cleaning techniques were applied, including:

Removal of special characters and numbers
Conversion to lowercase
Tokenization
Stopword removal
Lemmatization
3.5 Feature Transformation
The TF-IDF vectorizer is used to convert the text data into numerical features.

3.6 Model Building
Multinomial Naive Bayes (MNB)
Decision Tree
Random Forest
3.7 Model Evaluation
Cross-validation is employed to evaluate models using the F1-score. The performance of each model is assessed using classification reports and confusion matrices.

3.8 Model Comparison and Selection
After evaluating all models, the Random Forest model is selected as the best performer based on its F1-score and overall performance.

3.9 Enhanced Methodology with Deeper Learning Models
A neural network model using word embeddings and recurrent neural networks (RNN) is also explored, comparing its performance with traditional models.

3.10 Data Flow Diagram
A diagram illustrating the data flow and model architecture is provided.

4. Results
4.1 Traditional Machine Learning Models
Multinomial Naive Bayes: F1-Score = 0.943
Decision Tree: F1-Score = 0.98
Random Forest: F1-Score = 0.995
4.2 Deeper Learning Model
Word Embeddings + RNN: F1-Score = 0.981
5. Conclusion
The Random Forest model outperforms other models in terms of classification accuracy, making it the best choice for SMS spam detection. The neural network approach also shows promise, particularly in capturing more complex patterns in text data.

5.1 Key Findings
Random Forest showed superior performance.
Feature engineering and data cleaning were crucial for improving model accuracy.
Neural networks offer competitive results, though traditional models are more practical.
6. Future Work
Future enhancements could focus on:

Advanced Neural Network Architectures: Exploring models like DRNN or transformer-based models.
Hyperparameter Tuning: Fine-tuning the models for better performance.
Ensemble Methods: Combining different models to improve results.
Real-time Deployment: Implementing the model in real-world applications.
7. References
Olubayo, O., & Adebayo, A. A. (2022). "A survey of text message spam filtering techniques."
Jain, A., Kapoor, S., & Sarje, A. K. (2021). "SMS Spam Filtering: A Survey."
Milani, A. A., & Pinheiro, M. C. (2022). "A Comparative Analysis of SMS Spam Filtering Techniques."
Islam, M. F., & Islam, S. (2020). "Machine Learning-Based SMS Spam Detection."
