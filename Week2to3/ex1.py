import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

df = pd.DataFrame(
    {
        'name': ['Ali', 'Siti', 'John', 'Mei'],
        'age': [25, None, 30, 22],
        'salary': [5000, 6000, np.nan, 4500],
    }
)

print(df.isnull())
print(df.isnull().sum())
print(df.isnull().values.any())

complete_customers = df.dropna()

print(f"\nAfter removing incomplete records:")
print(f"Remaining customer: {complete_customers.shape[0]}")
print(complete_customers)

print(df['salary'].skew())

df['log_salary'] = np.log(df['salary'] + 1)

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# To display the chart
#plt.show()
