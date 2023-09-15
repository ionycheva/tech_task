import pandas as pd
import numpy as np

# Data loading
df = pd.read_excel('tmp.xlsx', header=1, )

# removing the ruble symbol and non-printing characters
df['Loan issued'] = pd.to_numeric(df['Loan issued'].str.replace('₽', '').str.replace('\u00A0', ''))
df['Earned interest'] = pd.to_numeric(df['Earned interest'].str.replace('₽', '').str.replace('\u00A0', ''))
df['Unpaid,  full amount'] = pd.to_numeric(
    df['Unpaid,  full amount'].str.replace('₽', '').str.replace('\u00A0', ''))

# casting some columns to percentage format
df['Comission, %'] = df['Comission, %'] * 100
df['EL'] = df['EL'] * 100
df = df.drop(columns=['Unnamed: 6'], axis=1)

# column definition
df = df[['Comission, %', 'Rating', 'Loan issued', 'Earned interest', 'Unpaid,  full amount', 'EL']]

# Making pivot table
df_pivot_table = df.pivot_table(index=['Comission, %'], columns=['Rating'],
                                aggfunc={'sum', 'mean'}, values=['Loan issued']).fillna(0)
df_pivot_table.to_excel('pivot_table.xlsx')  # saves results to xlsx-file

# An alternative pivot table that is more informative for me.
df_pivot_table1 = df.pivot_table(index=['Rating', 'Comission, %'],
                                 aggfunc={'sum', 'mean', 'max', 'min'},
                                 values=['Loan issued', 'Earned interest']).fillna(0)

df_pivot_table1.to_excel('alt_pivot_table.xlsx')  # saves results to xlsx-file
