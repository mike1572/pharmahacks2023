
# Code challenge PharmaHacks

import pandas as pd
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn import metrics


df_train = pd.read_excel('./train_data.xlsx', sheet_name=['x','y', 'labels']) 
df_test = pd.read_excel('./test_data_corrected.xlsx', sheet_name=['x','y', 'labels']) 


def addColumn(df): 
    df['x']['dd5 2nd deriv'] = (((df['x']['dd7 Glucose Concentration'] - df['x']['dd5 Glucose Concentration'])/2) - ((df['x']['dd5 Glucose Concentration'] - df['x']['dd3 Glucose Concentration'])/2)) /4

#addColumn(df_train)
#addColumn(df_test)

#print(df_test['y']['dd10 CM Content'])

df_train['y']['dd10 CM Content']  = [1 if x >= 90 else 0 for x in df_train['y']['dd10 CM Content']]
df_test['y']['dd10 CM Content']  = [1 if x >= 90 else 0 for x in df_test['y']['dd10 CM Content']]


'''
# Feature Selection

lr = LogisticRegression(max_iter=5000)
rfe = RFE(lr, n_features_to_select=3)
model = rfe.fit(df_train['x'], df_train['y'])
#model.ranking_
coef = pd.DataFrame(list(zip(df_train['x'].columns,model.ranking_)), columns = ['predictor','ranking'])

coefficients_df = coef.loc[coef['ranking']>=40]
predictors_to_keep = coefficients_df['predictor'].tolist()
df_train['x'] = df_train['x'][predictors_to_keep]
df_test['x'] = df_test['x'][predictors_to_keep]


model = Lasso(alpha=0.1)
model.fit(df_train['x'],df_train['y'])
coefficients = pd.DataFrame(list(zip(df_train['x'].columns,model.coef_)), columns = ['predictor','coefficient'])
coefficients_df = coefficients.loc[ abs(coefficients['coefficient']) > 0.9 ]
#coefficients_df = coefficients.loc[:, ((coefficients['coefficient']>=0.5) || (coefficients['coefficient']<=0.5)).any()]

predictors_to_keep = coefficients_df['predictor'].tolist()
df_train['x'] = df_train['x'][predictors_to_keep]
df_test['x'] = df_test['x'][predictors_to_keep]


randomforest = RandomForestClassifier(random_state=0)
model = randomforest.fit(df_train['x'], df_train['y'])
print(model.feature_importances_)
print(pd.DataFrame(list(zip(df_train['x'].columns,model.feature_importances_)), columns =
['predictor’,’feature importance']))


'''




def dropColumns(df): 
    df['x'] = df['x'].drop(columns=[
    #'dd0-dd1 Cell Density Gradient',
    'dd1-dd2 Cell Density Gradient',
    'dd2-dd3 Cell Density Gradient',
    'dd3-dd5 Cell Density Gradient',
    'dd5-dd7 Cell Density Gradient',
    'dd0-dd1 Aggregate Size Gradient',
    'dd1-dd2 Aggregate Size Gradient',
    'dd2-dd3 Aggregate Size Gradient',
    'dd3-dd5 Aggregate Size Gradient',
    'dd5-dd7 Aggregate Size Gradient',
    'Average DO concentration gradient d0',
    'Average DO concentration gradient d1',
    'DO gradient/cell count dd0',
    'DO gradient/cell count dd1',
    'DO gradient/cell count dd2',
    'DO gradient/cell count dd3',
    'DO gradient/cell count dd5',
    'DO gradient/cell count dd7',
    'dd0 Average of 2nd derivative DO',
    'dd1 Average of 2nd derivative DO',
    'dd2 Average of 2nd derivative DO', 
    'dd3 Average of 2nd derivative DO',
    'dd5 Average of 2nd derivative DO',
    'dd7 Average of 2nd derivative DO',
    'dd0 DO 2nd derivative/cell count',
    'dd1 DO 2nd derivative/cell count',
    
    # Better Recall, Worst Precision
    #'dd2 DO 2nd derivative/cell count',
    
    'dd3 DO 2nd derivative/cell count',
    'dd5 DO 2nd derivative/cell count',
    'dd7 DO 2nd derivative/cell count',
    'Overall density gradient',
    'Overall aggregate size gradient',
    'd0 Average pH Gradient',
    'd1 Average pH Gradient',
    'dd0 Average pH Gradient',
    'dd1 Average pH Gradient',
    'dd2 Average pH Gradient',
    'dd3 Average pH Gradient',
    'dd4 Average pH Gradient',
    'dd5 Average pH Gradient',
    'dd6 Average pH Gradient',
    'dd7 Average pH Gradient',
    'dd0 Lactate Concentration',
    'dd1 Lactate Concentration',
    'dd3 Lactate Concentration',
    'dd5 Lactate Concentration',
    'dd7 Lactate Concentration',
    'Preculture Time [h]',
    'Start Preculture Perfusion [h after inoc] d1-d2',
    'Presence of  IWP2 [h]',
    'Average DO concentration d0',
    ])


dropColumns(df_train)
dropColumns(df_test)

# Standardize the data
scaler = StandardScaler()
#scaler = MinMaxScaler()
X_train_std = scaler.fit_transform(df_train['x'])
X_test_std = scaler.fit_transform(df_test['x'])


# Loop to test for optimal paramters that will give us the highest accuracy
for i in range(2, 3):
    
    algo = RandomForestClassifier(
        random_state=10,
        max_features=5,
        n_estimators=100, 
        max_depth=20, 
        min_samples_split=2, 
        min_samples_leaf=1,
    )
    
  
    #algo = GradientBoostingClassifier(random_state=i,min_samples_split=5,n_estimators=150)
    #algo = MLPClassifier(hidden_layer_sizes=(35),max_iter=1000, random_state=5)
    
    model1 = algo.fit(X_train_std, df_train['y'])
    y_test_pred = model1.predict(X_test_std)
    mse = mean_squared_error(df_test['y'], y_test_pred)
    print(i)

    
    accuracy = metrics.accuracy_score(df_test['y']['dd10 CM Content'], y_test_pred)
    precision = metrics.precision_score(df_test['y']['dd10 CM Content'], y_test_pred,zero_division=0.0)
    recall = metrics.recall_score(df_test['y']['dd10 CM Content'], y_test_pred)
    matthews_coefficient = metrics.matthews_corrcoef(df_test['y']['dd10 CM Content'], y_test_pred)
    confusion_matrix = metrics.confusion_matrix(df_test['y']['dd10 CM Content'], y_test_pred)
    
    print("Including predictor dd2 DO 2nd derivative/cell count")
    print("Mean Squared Error: ", mse)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Matthews Correlation Coefficient: ", matthews_coefficient)
    print("Confusion Matrix: \n", confusion_matrix)
        
    # Plot the Confusion Matrix
    
    # Define the data
    values = [confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0], confusion_matrix[1][1]]
    
    # Define the labels for the x-axis
    labels = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
    
    # Create the bar chart
    fig, ax = plt.subplots()
    ax.bar(labels, values, color='red')
    
    # Add a title
    ax.set_title("Confusion Matrix With 'dd2 DO 2nd derivative/cell count' Excluded as Predictor") 
    # Add labels to the axes
    ax.set_ylabel('Amount')
   

    # Show the plot
    plt.show()

    
    
    

'''

# Simple Linear Regression
lm = LinearRegression()
model1 = lm.fit(X_train_std, df_train['y'])
y_pred = model1.predict(X_test_std)
mse1 = mean_squared_error(df_test['y'], y_pred)
print("Linear Regression: ", mse1)
x = df_test['x']
y = df_test['y']


# Lasso Regression
lasso = Lasso(alpha=1)
model2 = lasso.fit(X_train_std, df_train['y'])
y_test_pred = model2.predict(X_test_std)
mse2 = mean_squared_error(df_test['y'], y_test_pred)
print("Lasso: ", mse2)


# Gradient Boosting Regressor
gbt = GradientBoostingRegressor(random_state=0, n_estimators=200)
model4 = gbt.fit(X_train_std, df_train['y'])
y_test_pred = model4.predict(X_test_std)
mse4 = mean_squared_error(df_test['y'], y_test_pred)
print("Gradient Boosting Regressor: ", mse4)


# Artificial Neural Network
ann = MLPRegressor(hidden_layer_sizes=(1),max_iter=1000,random_state=0)
model5 = ann.fit(X_train_std, df_train['y'])
y_test_pred = model5.predict(X_test_std)
mse5 = mean_squared_error(df_test['y'], y_test_pred)
print("ANN: ", mse5)


# Random Forest Regressor
for i in range(2, 3): 
    rf = RandomForestRegressor(n_estimators=59, max_depth=10, min_samples_split=2, min_samples_leaf=8, max_features=29, random_state=20)
    model3 = rf.fit(X_train_std, df_train['y'])
    y_test_pred = model3.predict(X_test_std)
    mse3 = mean_squared_error(df_test['y'], y_test_pred)
    print("Random Forest Regressor MSE: ", mse3)
    #print(i)


columns = ['testing', 'prediction']

y_testing_classify = pd.DataFrame(columns=columns)
y_testing_classify['testing'] = [1 if x >= 90 else 0 for x in df_test['y']['dd10 CM Content']]
y_testing_classify['prediction'] = [1 if x >= 90 else 0 for x in y_test_pred]


accuracy = metrics.accuracy_score(y_testing_classify['testing'], y_testing_classify['prediction'])
precision = metrics.precision_score(y_testing_classify['testing'], y_testing_classify['prediction'],zero_division=0.0)
recall = metrics.recall_score(y_testing_classify['testing'], y_testing_classify['prediction'])
matthews_coefficient = metrics.matthews_corrcoef(y_testing_classify['testing'], y_testing_classify['prediction'])


print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("Matthews Correlation Coefficient: ", matthews_coefficient)

'''
