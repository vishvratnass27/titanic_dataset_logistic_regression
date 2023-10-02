# titanic_dataset_logistic_regression

# Titanic Survival Prediction

This project aims to predict the survival of passengers on the Titanic using various machine learning algorithms. The dataset used is the famous Titanic dataset, which contains information about passengers such as age, gender, class, and whether they survived or not.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The sinking of the Titanic is one of the most infamous shipwrecks in history. This project uses machine learning to analyze the dataset and predict which passengers were likely to survive based on various features.

## Dependencies

- Python 3
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

You can install the required packages using pip:

```
pip install pandas matplotlib seaborn scikit-learn
```

## Dataset

The dataset used in this project is named `train.csv`. It includes the following columns:

- `PassengerId`: A unique identifier for each passenger.
- `Survived`: 1 if the passenger survived, 0 if not.
- `Pclass`: The passenger's class (1st, 2nd, or 3rd).
- `Name`: The passenger's name.
- `Sex`: The passenger's gender.
- `Age`: The passenger's age.
- `SibSp`: The number of siblings/spouses aboard.
- `Parch`: The number of parents/children aboard.
- `Ticket`: The ticket number.
- `Fare`: The ticket fare.
- `Cabin`: The cabin number.
- `Embarked`: The port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## Data Preprocessing

- Removed unnecessary columns (`PassengerId`, `Name`, `Ticket`, `Fare`, `Parch`, `Cabin`).
- Handled missing values in the `Age` column.
- Converted categorical variables (`Pclass`, `Sex`, `SibSp`, `Embarked`) into numerical format using one-hot encoding.

## Exploratory Data Analysis (EDA)

Performed EDA to understand the relationships between features and survival, including visualizations like countplots and heatmaps.

## Modeling

Utilized several machine learning algorithms, including:

- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)

## Evaluation

Evaluated model performance using:

- Confusion Matrix
- Accuracy Score

