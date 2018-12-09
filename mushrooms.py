#import all the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error 

# Load the dataset
data = pd.read_csv(r'path-to-the-directory\mushrooms.csv')
# Print the starting 5 samples
print(data.head())

# The dataset contain the non-numerical data. covert the data to the format supported by the algorithms 
# This data covertion method is called data wrangling
# For the features having 2 different entries the we will map one entry to 1 and the other one to 0
# For the features having more than 2 types of entries we create more columns equal to the number of entries with the prefix being...
# ... old column name and the entries as 1 or 0 according to the type
# Ex: for a column X with types a,b,c the column will be split as X_a,X_b,X_c and X_will be 1 for the wherever the entries in the old..
# column was a and rest entries in the X_a will be 0 same for X_b and X_c

def wrangle(data):

    data['bruises'] = data['bruises'].map({'t': 1, 'f':0}).astype(int)
    
    cap_shape_sep = pd.get_dummies(data['cap-shape'],prefix = 'cap-shape')
    data = pd.concat([data,cap_shape_sep],axis=1)
    data = data.drop('cap-shape',axis=1)
    
    cap_surf_sep = pd.get_dummies(data['cap-surface'],prefix = 'cap-surface')
    data = pd.concat([data,cap_surf_sep],axis=1)
    data =  data.drop('cap-surface',axis=1)

    cap_color_sep = pd.get_dummies(data['cap-color'],prefix = 'cap-color')
    data = pd.concat([data,cap_color_sep],axis=1)
    data =  data.drop('cap-color',axis=1)
    
    odor_sep = pd.get_dummies(data['odor'],prefix = 'odor')
    data = pd.concat([data,odor_sep],axis=1)
    data =  data.drop('odor',axis=1)
    
    data['gill-attachment'] = data['gill-attachment'].map({'f': 1, 'a':0}).astype(int)
    
    data['gill-spacing'] = data['gill-spacing'].map({'c': 1, 'w':0}).astype(int)
    
    data['gill-size'] = data['gill-size'].map({'n': 1, 'b':0}).astype(int)
    
    gill_color_sep = pd.get_dummies(data['gill-color'],prefix = 'gill_color')
    data = pd.concat([data,gill_color_sep],axis=1)
    data =  data.drop('gill-color',axis=1)
    
    data['stalk-shape'] = data['stalk-shape'].map({'e': 1, 't':0}).astype(int)
    
    stalk_root_sep = pd.get_dummies(data['stalk-root'],prefix = 'stalk_root')
    data = pd.concat([data,stalk_root_sep],axis=1)
    data =  data.drop('stalk-root',axis=1)

    stalk_surf_above_sep = pd.get_dummies(data['stalk-surface-above-ring'],prefix = 'stalk-surface-above-ring')
    data = pd.concat([data,stalk_surf_above_sep],axis=1)
    data =  data.drop('stalk-surface-above-ring',axis=1)

    stalk_surf_below_sep = pd.get_dummies(data['stalk-surface-below-ring'],prefix = 'stalk-surface-below-ring')
    data = pd.concat([data,stalk_surf_below_sep],axis=1)
    data =  data.drop('stalk-surface-below-ring',axis=1)

    stalk_color_above_sep = pd.get_dummies(data['stalk-color-above-ring'],prefix = 'stalk-color-above-ring')
    data = pd.concat([data,stalk_color_above_sep],axis=1)
    data =  data.drop('stalk-color-above-ring',axis=1) 
    
    stalk_color_below_sep = pd.get_dummies(data['stalk-color-below-ring'],prefix = 'stalk-color-below-ring')
    data = pd.concat([data,stalk_color_below_sep],axis=1)
    data =  data.drop('stalk-color-below-ring',axis=1)

    veil_color_sep = pd.get_dummies(data['veil-color'],prefix = 'veil-color')
    data = pd.concat([data,veil_color_sep],axis=1)
    data =  data.drop('veil-color',axis=1)   
    
    ring_num_sep = pd.get_dummies(data['ring-number'],prefix = 'ring-number')
    data = pd.concat([data,ring_num_sep],axis=1)
    data = data.drop('ring-number',axis=1)
    
    ring_type_sep = pd.get_dummies(data['ring-type'],prefix='ring-type')
    data= pd.concat([data,ring_type_sep],axis=1)
    data = data.drop('ring-type',axis=1)
    
    spore_print_color_sep = pd.get_dummies(data['spore-print-color'],prefix = 'spore-print-color')
    data = pd.concat([data,spore_print_color_sep],axis=1)
    data = data.drop('spore-print-color',axis=1)
    
    population_sep = pd.get_dummies(data['population'],prefix = 'population')
    data = pd.concat([data,population_sep],axis=1)
    data = data.drop('population',axis=1)
    
    habitat_sep = pd.get_dummies(data['habitat'],prefix = 'habitat')
    data = pd.concat([data,habitat_sep],axis=1)
    data = data.drop('habitat',axis=1)

# class column contains the labels(edible or poisonous) so we seperate the column
    data['class'] = data['class'].map({'e': 1, 'p':0}).astype(int)
    data = data.drop('veil-type',axis=1)
    return data
    
data = wrangle(data)
print(data.info())
print(data.head())

Y = data['class']
X = data.drop('class',axis=1)

# create the validation set from the given data
X_train, X_val, y_train, y_val = train_test_split(X, Y, 
                                                    test_size=0.1, 
                                                    random_state=123, 
                                                    stratify=Y)

# split the remaining data into test and training dataset
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 
                                                    test_size=0.1, 
                                                    random_state=123, 
                                                    stratify=y_train)

# Choose a classifer model
clf = RandomForestClassifier()

# make sure all the features are in the right forma
print(X_train.dtypes)

# train the selected classifier with training data set
clf.fit(X_train,y_train)

# evaluate the classifier on training and validation set
train_acc = round(clf.score(X_train,y_train),2)
val_acc = round(clf.score(X_val,y_val),2)
print('training_acuracy:',train_acc)
print('val_accuracy:',val_acc)

# check the decision making ability of the model on unseen test data
y_pred = clf.predict(X_test)

# print the starting 10 samples of predicted and actual y test
print('predicted',y_pred[:10])
print('y_test',y_test[:10])

# check how the actual and predicted values differ from each other
err = mean_squared_error(y_pred,y_test)
print('error:',err)
