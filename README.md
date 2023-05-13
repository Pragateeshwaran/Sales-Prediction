# Advertising Sales Prediction
This project aims to predict sales based on advertising spend for a company. The goal is to identify which advertising channels (TV, radio, newspaper) have the biggest impact on sales, and how much money should be allocated to each channel to maximize sales.

# Data Description
The data used in this project comes from a fictional company that tracked their advertising spend and sales over a period of time. The dataset includes 200 observations and 4 variables: TV, radio, newspaper, and sales. There was no missing data in the dataset.

# Machine Learning Model
The machine learning model used in this project is linear regression. The purpose of the model is to predict sales based on the advertising spend in TV, radio, and newspaper. The accuracy of the model was evaluated using the R-squared metric, which was found to be 0.91 on the test set.

# Code Snippets
The following code snippets were used in the project:

Reading the data from a CSV file and loading it into a Pandas dataframe:
python
Copy code

    import pandas as pd

    df = pd.read_csv("advertising.csv")
    
Splitting the data into training and test sets:
python
Copy code
    
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
Training a linear regression model and evaluating its accuracy:
python
Copy code
        
      from sklearn.linear_model import LinearRegression
      model = LinearRegression()
      model.fit(X_train, y_train)
      score = model.score(X_test, y_test
      
# Running the Project
To run the project, you will need to have Python 3 and the following packages installed: pandas, numpy, seaborn, and scikit-learn. Once you have installed these packages, you can run the project by opening the Jupyter notebook and running each cell in order.

# Acknowledgements
I would like to thank the fictional company that provided the dataset for this project.

# License
This project is licensed under the MIT License. See LICENSE.md for more information.
