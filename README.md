## Housing Price Prediction Model

This repository contains code for a machine learning model to predict housing prices based on various features. The model uses different algorithms such as Linear Regression, Decision Tree Regression, LassoCV, and RidgeCV to predict the prices.

### Prerequisites
- Python 3
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn

### Dataset
The dataset used for this project is named `housing.csv`. It contains the following features:

1. CRIM: Per capita crime rate by town
2. ZN: Proportion of residential land zoned for lots over 25,000 sq. ft.
3. INDUS: Proportion of non-retail business acres per town
4. CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
5. NOX: Nitric oxides concentration (parts per 10 million)
6. RM: Average number of rooms per dwelling
7. AGE: Proportion of owner-occupied units built prior to 1940
8. DIS: Weighted distances to five Boston employment centers
9. RAD: Index of accessibility to radial highways
10. TAX: Full-value property tax rate per $10,000
11. PTRATIO: Pupil-teacher ratio by town
12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
13. LSTAT: Percentage lower status of the population
14. MEDV: Median value of owner-occupied homes in $1000's (target variable)

### Usage
1. Clone this repository.
2. Ensure Python and required libraries are installed.
3. Place the `housing.csv` file in the same directory as the code.
4. Run the code.

### Code Explanation
- Import necessary libraries: Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn modules.
- Load the dataset using Pandas.
- Explore the data using `head()`, `info()`, and `describe()` functions.
- Visualize the correlation matrix using a heatmap.
- Split the dataset into training and testing sets.
- Train and evaluate the Linear Regression model.
- Train and evaluate the Decision Tree Regression model.
- Train and evaluate the LassoCV model for regularization.
- Train and evaluate the RidgeCV model for regularization.

### Results
The code generates scatter plots to visualize the predicted prices against the actual prices for each model. It also prints the training and testing accuracies along with the model accuracy (R2 score).

### Conclusion
This README provides an overview of the housing price prediction model and instructions on how to use the code. For further details, refer to the code comments and documentation of the libraries used.
