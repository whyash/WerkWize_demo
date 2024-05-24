import numpy as np
import pandas as pd
from scipy import stats

data = pd.read_csv('Devtypes.csv',nrows=1000)
print(data.head())

contingTab = pd.crosstab(data.Role, data.Type, margins=True)

print(contingTab)

print(contingTab['INFP'])

print(contingTab.transpose())

roles = list(data['Role'].unique())
types = list(data['Type'].unique())

exp1 = {}

for i in roles:
  exp2 = {}
  for j in types:
    exp2[j] = contingTab.transpose()[i]['All'] * contingTab[j]['All'] / (contingTab['All']['All'])

  exp1[i] = exp2

print(exp1)

dof = (len(roles)-1) * (len(types)-1)
print(dof)

chiSquareCal = 0
for i in roles:
  for j in types:
    val = (contingTab.transpose()[i][j] - exp1[i][j])**2/exp1[i][j]
    chiSquareCal = chiSquareCal + val

print(stats.chi2.ppf(1-0.075, df=dof))

contab = np.array([contingTab.transpose()['Front End Developer'][0:16].values,
                  contingTab.transpose()['Web Developer'][0:16].values,
                   contingTab.transpose()['Software Engineer'][0:16].values,
                   contingTab.transpose()['Computer Engineer'][0:16].values])

print(stats.chi2_contingency(contab))
print(1 - stats.chi2.cdf(chiSquareCal, dof)) 


