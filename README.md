# titanic_dataset_logistic_regression

This code is for a simple machine learning project that predicts whether passengers on the Titanic survived or not based on various features. Here's a step-by-step explanation of the code:

1. **Import Libraries**: Import necessary libraries like pandas, matplotlib, seaborn, numpy, and scikit-learn.

2. **Load the Dataset**: Load the Titanic dataset from a CSV file named "train.csv" into a DataFrame called `dataset`.

3. **Data Exploration**:
   - Use `dataset.head()` to display the first few rows of the dataset.
   - Use `dataset.columns` to list the column names in the dataset.

4. **Data Visualization**:
   - Visualize the impact of some columns on survival using seaborn countplots, such as 'Sex', 'Pclass', 'SibSp', 'Parch', and 'Embarked'. These plots show how the distribution of passengers who survived or didn't vary based on these features.

5. **Data Cleaning**:
   - Remove columns that are unlikely to have a significant impact on survival, like 'PassengerId', 'Name', 'Ticket', 'Fare', 'Parch', and 'Cabin'.
   - Check for missing values in the dataset using `dataset.isnull()`.
   - Use a heatmap from seaborn (`sns.heatmap`) to visualize missing values in the dataset. Lighter areas indicate missing values.
   - Calculate mean ages for different passenger classes ('Pclass') and fill missing age values based on these means.

6. **One-Hot Encoding**:
   - Convert categorical variables like 'Pclass', 'Sex', 'SibSp', and 'Embarked' into binary (0 or 1) columns using one-hot encoding.

7. **Data Preparation**:
   - Separate the target variable ('Survived') from the dataset and store it in the variable `y`.
   - Remove the 'Survived' column from the dataset (`dataset.drop("Survived", axis=1)`).
   - Store the remaining features in the variable `x`.

8. **Logistic Regression**:
   - Import `LogisticRegression` from scikit-learn.
   - Split the data into training and testing sets using `train_test_split`.
   - Create a logistic regression model (`model`) and fit it to the training data.
   - Use the trained model to make predictions on the test data (`x_test`) and store the predictions in `y_pred`.

9. **Evaluation**:
   - Calculate the confusion matrix using `confusion_matrix` to measure the number of true positives, true negatives, false positives, and false negatives.
   - Calculate the accuracy of the model using `accuracy_score` from scikit-learn.

The code performs data exploration, data cleaning, feature engineering, and model training using logistic regression to predict survival on the Titanic. The accuracy of the model is also evaluated.
