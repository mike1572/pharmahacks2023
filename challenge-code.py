
# Code challenge PharmaHacks

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
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
    'Average DO concentration gradient dd0',
    'Average DO concentration gradient dd1',
    'Average DO concentration gradient dd2',
    'Average DO concentration gradient dd3',
    'Average DO concentration gradient dd4',
    'Average DO concentration gradient dd5',
    'Average DO concentration gradient dd6',
    'Average DO concentration gradient dd7',
    'DO concentration/cell count dd0',
    'DO concentration/cell count dd1',
    'DO concentration/cell count dd2' ,
    'DO concentration/cell count dd3',
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
    'Overall Average pH',
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
    'Presence of  IWP2 [h]'
    ])


dropColumns(df_train)
dropColumns(df_test)

#scaler = StandardScaler()
#X_std = scaler.fit_transform(df_train['x'])

lm = LinearRegression()
model1 = lm.fit(df_train['x'], df_train['y'])

#X_train, X_test, y_train, y_test = train_test_split(df_train['x'], df_train['y'], test_size=0.2, random_state=0) 

#y_test_pred = model1.predict(X_test)
y_pred = model1.predict(df_test['x'])

#mse1 = mean_squared_error(y_test, y_pred)
mse1 = mean_squared_error(df_test['y'], y_pred)
print(mse1)
x = df_test['x']
y = df_test['y']

print(metrics.accuracy_score(df_test['y'], y_pred))
print(metrics.confusion_matrix(df_test['y'], y_pred))

'''
plt.figure(figsize=(10,6))


print(x.shape[0])
print(y.shape[0])

plt.scatter(x=x['Presence of  IWP2 [h]'], y=y, color='g')
plt.plot(df_test['x']['Presence of  IWP2 [h]'], y_pred,color='k') 
plt.show()
'''