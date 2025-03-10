import pandas as pd
import numpy as np

df=pd.read_csv('Cleaned.csv')
to_drop = ['Projects Count','Recruiter Decision','AI Score (0-100)','Job Role','Certifications']

df.drop(to_drop, inplace=True, axis=1)
print(df)
