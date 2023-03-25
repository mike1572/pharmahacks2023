
# Code challenge PharmaHacks

import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn import metrics

#df = pandas.read_csv("./train_data.xlsx")
df_train = pd.read_excel('./train_data.xlsx', sheet_name=['x','y', 'labels']) 
df_test = pd.read_excel('./test_data_corrected.xlsx', sheet_name=['x','y', 'labels']) 

def dropColumns(df): 
    df['x'] = df['x'].drop(columns=[
    'dd0-dd1 Cell Density Gradient',
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
    'dd2 DO 2nd derivative/cell count',
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
    'Average DO concentration d0'
    ])


dropColumns(df_train)
dropColumns(df_test)


# Standardize the data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(df_train['x'])
X_test_std = scaler.fit_transform(df_test['x'])


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

# Random Forest Regressor
rfr = RandomForestRegressor(max_depth=(10), random_state=1, n_estimators=250)
model3 = rfr.fit(X_train_std, df_train['y'])
y_test_pred = model3.predict(X_test_std)
mse3 = mean_squared_error(df_test['y'], y_test_pred)
print("Random Forest Regressor: ", mse3)


# Gradient Boosting Regressor
gbt = GradientBoostingRegressor(random_state=0, n_estimators=200)
model4 = gbt.fit(X_train_std, df_train['y'])
y_test_pred = model4.predict(X_test_std)
mse4 = mean_squared_error(df_test['y'], y_test_pred)
print("Gradient Boosting Regressor: ", mse4)


# Artificial Neural Network
ann = MLPRegressor(hidden_layer_sizes=(15),max_iter=100000,random_state=0)
model5 = ann.fit(X_train_std, df_train['y'])
y_test_pred = model5.predict(X_test_std)
mse5 = mean_squared_error(df_test['y'], y_test_pred)
print("ANN: ", mse5)


model = Lasso(alpha=0.01)
model.fit(X_train_std,df_train['y'])
model.coef_
#pd.DataFrame(list(zip(X_train_std.columns,model.coef_)), columns = ['predictor','coefficient'])




plt.figure(figsize=(10,6))


print(x.shape[0])
print(y.shape[0])

#plt.scatter(x=y_test_pred, y=df_test['y'], color='g')
#plt.plot(x['Overall Average pH'], y_pred,color='k') 
plt.scatter(df_test['y'].index, df_test['y'].values, color='k')
plt.scatter(y_test_pred.index, y_test_pred.values, color='h')
plt.show()


