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
     
     get_dummies() creates dummy columns with with name having the prefix given by the user. EX: for a column X with types a,b,c the          column will be split as X_a,X_b,X_c and X_a will be 1 wherever the entries in the old column was a and rest entries in the X_a will      be 0 same for X_b and X_c 
     
     concat() concates the new columns with the dataset
     
     drop() drops the old column.

Line 103 : info() print the information about all the features in the dataset

Line 104 : head() print the 5 rows of dataset

Line 110,116 : train_test_split() is a method in sklearn.model_selection splits the dataset into training and test datasets, this method                splits the dataset in such a way that each section has uniform distribution of all classes [more about train test split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) 
             This method is called twice to devide the dataset as training, validation and test sets.
 
We can always visualise how the feature and the output are related and desirable changes can be made to the feature before feeding it to the training model, if there is no relation than that feature can be dropped,

Line 122 : Classifier used is RandomForestClassifier method from sklearn.ensemble [All about Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)

Line 131,132 : score() evaluates the trained model performance on both training and validation data set [evaluation methods available in sklearn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)

Line 137 : predict() makes predictions on the test data.

Line 140,141 : print the 10 samples of actual and predicted output

Line 145 : print the difference between predicted and actual data calculated using mean_suared_error() from sklearn.metrics 

The output of the RandomForestClassifier funstion:
* train_accuracy = 1.0
* val_accuracy = 1.0
* mean_squared_error between predicted and actual Y_test = 0.0070

* The output of the code is as follows:
* Training accuracy : 1
* validation set accuracy : 1

The model works well for both the training data and test data.
