
import pandas as pd
import numpy as np


# Sample DataFrame
# data = np.loadtxt('data/bank/train.csv', dtype=str)
data = pd.read_csv('data/bank/train.csv', header=None)
print(data)
# print(frame)
df = pd.DataFrame(data)
# print(df)

# Iterate through the columns
for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        median = df[column].median()
        df[column] = (df[column] > median).astype(int)

# print(df)
for column in df.columns:
    print(column)
