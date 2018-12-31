# mushrrom-classification
Machine learning code to decide whether the mushroom is edible or poisonous based on the different features of mushroom
 
!!! CODE INSIGHT!!!

Line 2-6 : Import all the required libraries and modules 
       [More datails about the SKlearn library](https://scikit-learn.org/stable/user_guide.html)

Download the dataset [hear](https://www.kaggle.com/uciml/mushroom-classification)
Line 9  : Load the downloaded dataset with pandas read_csv() method
       [All about Pandas](https://pandas.pydata.org/pandas-docs/stable/)

Data wrangling is the method used to convert the features in the data set into the format supported by the algorithms. Ex: Converting the features of datatype strings into intiger or float.

Line 21 : wrangle() performs data wrangling
          
  Two methods are used in  the code.
          
  1. map function : Used for the features with only two unique entries. This method maps one entry as 1 and another as 0.
  
  2. When a feature has more than two unique entries.
     get_dummies() creates dummy columns with with name having the prefix given by the user 
Line 38-43 : train_test_split() is a method in sklearn.model_selection splits the dataset into training and test datasets, this method                splits the dataset in such a way that each section has uniform distribution of all classes [more about train test split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) 
             This method is called twice to devide the dataset as training, validation and test sets.
 
We can always visualise how the feature and the output are related and desirable changes can be made to the feature before feeding it to the training model, if there is no relation than that feature can be dropped,

Line 55 : light() plots the gragh indicating the relation between feature light and the output, call it once to understand the relation

Line 68 : logistic() uses LogisticRegression method from sklearn.linear_model [All about linear models](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
The output of the Logistic funstion:
* train_accuracy = 0.99
* val_accuracy = 0.99
* mean_squared_error between predicted and actual Y_test = 0.0075

Line 85 : rf() uses RandomForestClassifier method from sklearn.ensemble [All about Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html) 
The output of the RandomForestClassifier funstion:
* train_accuracy = 1.0
* val_accuracy = 1.0
* mean_squared_error between predicted and actual Y_test = 0.0070

* The output of the code is as follows:
* Training accuracy : 1
* validation set accuracy : 1

The model works well for both the training data and test data.
