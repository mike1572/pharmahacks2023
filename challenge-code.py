
# Code challenge PharmaHacks


import pandas as pd


#df = pandas.read_csv("./train_data.xlsx")
df = pd.read_excel('./train_data.xlsx', sheet_name=['x','y', 'labels']) 

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
'DO concentration/cell count dd3'])






