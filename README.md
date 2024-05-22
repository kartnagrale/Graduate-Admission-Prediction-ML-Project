# Graduate-Admission-Prediction-ML-Project

The "Graduate Admission Prediction using Machine Learning" project presents a cutting-edge solution to enhance the graduate admission process, leveraging the power of Machine Learning (ML). With the ever-increasing competition for graduate program admissions, both applicants and admission committees seek more efficient and transparent decision-making processes. The project addresses this challenge by developing a predictive model that assesses the likelihood of a student's acceptance into a graduate program.

To achieve this, the project has harnessed a comprehensive dataset comprising historical admission records, standardized test scores, and academic performance indicators. Through the implementation of several ML algorithms, including Logistic Regression, Support Vector Machine (SVM), K-Nearest Neighbors Classifier (KNN), Random Forest Classifier, and Gradient Boosting Classifier, we have created a robust predictive framework. Each of these algorithms has been carefully selected and optimized to provide accurate predictions. Logistic Regression offers simplicity and interpretability, while SVM excels in handling non-linear data. KNN leverages proximity-based learning, Random Forest brings ensemble learning capabilities, and Gradient Boosting enhances model performance through boosting techniques.

# Work Done
We have used a combination of data, machine learning techniques, and statistical analysis to predict the likelihood of a student being admitted to a graduate program. We have distributed whole process in five modules. Here are the key components and steps used in the project:

** Module 1 ** 

Data Collection: Gathering historical data on past graduate applicants. This data include information such as GRE scores, undergraduate GPA, letters of recommendation, statement of purpose, and the admission outcomes (admitted or not admitted).

Data Preprocessing: Cleaning and preprocessing the data. This involves handling missing values, normalizing data, and converting categorical variables into numerical representations.

Feature Selection: Identifying and selecting the relevant features that are likely to influence the admission decision. We may also create new features through feature engineering, such as combining GRE scores and GPA into a composite score.

Feature Scaling: Standardize or normalize data to ensure all features are on a consistent scale.

** Module 2 **

Exploratory Data Analysis (EDA): Performing exploratory data analysis to understand the distribution of data, correlations between variables, and any patterns or insights that may be relevant for prediction.

Machine Learning Model Selection: Choosing the appropriate machine learning algorithms for your prediction task. Common choices include logistic regression, decision trees, random forests, support vector machines, and neural networks.

Data Splitting: Spliting the data into a training set and a testing set to evaluate the model's performance. Cross-validation may also be used to tune hyperparameters and assess model robustness.

** Module 3 **

Model Training using Regression methods: Training the selected machine learning model(s) on the training data.

We have used LinearRegression, Support Vector Regression, Random Forest Regressor, and Gradient Boosting Regressor based on dataset characteristics and project requirements for Regression method.

Linear regression is a simple and widely used statistical method for modeling the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship and aims to find the best-fit line that minimizes the sum of squared errors. Linear regression is interpretable and can be applied for predictive tasks when the relationship between variables is linear.

Support Vector Regression is a regression technique that extends the concepts of support vector machines (SVMs) to predict continuous values. It uses support vectors to define a margin of tolerance around the regression line, allowing for non-linear relationships between variables. SVR is effective when the relationship between variables is non-linear and can handle outliers.

Random Forest Regressor is an ensemble learning method that combines multiple decision trees to make predictions. It's a powerful model for both classification and regression tasks. In regression, it aggregates the predictions of individual decision trees to provide accurate and robust results. Random Forest is known for handling complex relationships and capturing feature importance.

Gradient Boosting Regressor is another ensemble technique that builds an ensemble of decision trees sequentially. It focuses on minimizing the residual errors from the previous tree, resulting in a strong predictive model. Gradient Boosting is particularly effective in capturing complex patterns and achieving high predictive accuracy in regression tasks.

** Module 4 **

Model Training using Classification methods: Training the selected machine learning model(s) on the training data.

We have used Support Vector Classifier, KNeighborsClassifier, RandomForestClassifier, GradientBoostingClassifier based on dataset characteristics and project requirements for Classification method.

Support Vector Classifier, often referred to as Support Vector Machine (SVM) for classification, is a powerful supervised learning algorithm. It classifies data points into different categories by finding the optimal hyperplane that maximizes the margin between classes. SVMs are effective in handling both linear and non-linear classification tasks and can be extended to handle multi-class classification as well.

The K-Nearest Neighbors Classifier is a simple and intuitive algorithm for classification. It classifies data points based on the majority class of their K nearest neighbors in the feature space. It's non-parametric and instance-based, making it effective for a wide range of classification problems, especially when local patterns matter.

The Random Forest Classifier is an ensemble learning method that combines multiple decision trees to make classification predictions. It's known for its robustness and high accuracy. Random Forest generates many decision trees and aggregates their results to provide a final classification. It's particularly effective for complex and high-dimensional datasets.

Gradient Boosting Classifier is an ensemble learning method that builds an ensemble of decision trees sequentially. It focuses on minimizing classification errors by adding trees that correct the mistakes of the previous ones. This iterative process results in a strong and accurate classification model. Gradient Boosting is suitable for a wide range of classification tasks and often outperforms other algorithms.

** Module 5 **

Model Evaluation: Evaluating the model's performance using appropriate metrics, such as accuracy, precision, recall, F1 score, and ROC AUC. You may also use methods like confusion matrices to assess model performance. In our project we have used R2 score for regression models and accuracy score metric for classification models.

R-squared or R2 is a statistical measure of how well regression predictions approximate real data points. It's also known as the coefficient of determination. R2 values range from 0 to 1. An R2 of 0 means that the model explains or predicts 0% of the relationship between the dependent and independent variables. An R2 of 1 indicates that the regression predictions perfectly fit the data.

Accuracy score is a classification measure in machine learning that represents the percentage of correct predictions made by a model. It is defined as the number of correct predictions divided by the total number of predictions, multiplied by 100.    

Model Validation: Validating the model on the testing dataset to ensure it generalizes well to new, unseen data.

Model Testing: In this, the trained machine learning models are put to the test in real-world scenarios. A separate dataset, not previously used for training or validation, is employed to assess how well the models perform when faced with unseen data. This testing stage aims to evaluate the models' ability to generalize and make accurate predictions beyond the data they were trained on.

Model Deployment: This involves the integration of the trained and validated machine learning models into the target application or system for practical use. The deployment process ensures that the models are accessible and operational, allowing stakeholders to make real-time predictions. It may involve setting up APIs, embedding the models in web or mobile applications, or integrating them into university admission systems. In this we have created a simple Graphical User Interface for the by deployment of the project into our GUI.            

The specific techniques and tools used in a graduate admission prediction project can vary depending on the scope and complexity of the project, as well as the available data and resources. Machine learning libraries like scikit-learn and deep learning frameworks also some python libraries like numpy , seaborn , pandas etc are often used in these projects. Additionally, ethical considerations and data privacy must be taken into account when working with sensitive admissions data.

![image](https://github.com/kartnagrale/Graduate-Admission-Prediction-ML-Project/assets/115936211/e2511633-3c61-4683-8b65-98fd046cd1ab)

** System Architecture **

# Results 

The "Graduate Admission Prediction using ML" project has successfully developed a predictive model that estimates the likelihood of admission for aspiring graduate students. The model takes into account various parameters, including GRE score, TOEFL score, university rating, and CGPA, and provides applicants with an outcome that categorizes their admission chances as either "High" or "Low."
 
![image](https://github.com/kartnagrale/Graduate-Admission-Prediction-ML-Project/assets/115936211/4c57409b-49d9-4a82-8cd3-0628086355eb)

Fig.1 R square scores of regression models

Notable R square scores for Regression Models are observed for Linear Regression (0.821208), Support Vector Regression (0.759781), Decision Tree (0.803149), Random Forest (0.797044).

![image](https://github.com/kartnagrale/Graduate-Admission-Prediction-ML-Project/assets/115936211/56b00c13-7b5d-4b7a-bdfc-826985b80e70)

Fig.2 Accuracy scores of classification models

Notable accuracy scores for Classifier Models are observed for Logistic Regession (0.9250), Support Vector (0.9250), KNeighbors (0.8875), Random Forest (0.9500), and Gradient Boosting (0.9750).

Upon extensive testing and validation, the model achieved a prediction accuracy of approximately 97%, which indicates its effectiveness in providing insights into the admission process. 

![image](https://github.com/kartnagrale/Graduate-Admission-Prediction-ML-Project/assets/115936211/05d4f696-fe95-4c2e-bfa5-d0e547ab9d80)

Fig.3 Interface for Graduate Admission Prediction

It is important to note that this accuracy is based on the data available for this study, and real-world performance may vary based on the specific dataset and criteria used by educational institutions.
