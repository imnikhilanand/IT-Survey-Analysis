# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 02:18:09 2021

@author: Nikhil
"""

"""
Spersman Correlation Coefficient

.00-.19 “very weak”
.20-.39 “weak”
.40-.59 “moderate”
.60-.79 “strong”
.80-1.0 “very strong”

"""

"""Importing the libraries"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

"""Reading the dataset"""
data_1 = pd.read_csv('Dataset/IT_Salary_Survey_EU_2018.csv')
data_2 = pd.read_csv('Dataset/IT_Salary_Survey EU_2019.csv')
data_3 = pd.read_csv('Dataset/IT_Salary_Survey EU_2020.csv')

"""For the first data frame"""

# deleting the timestamp column
del data_1['Timestamp']

# describing the data
data_1_describe = data_1.describe()

# drop rows with all the columns with null values
data_1 = data_1.dropna(how="all")

#heat map to check missing values
sns.heatmap(data_1.isnull(),cbar=False)

""" Imputing the values of Age """

""" 
First we have to check if we are able to find correlation through one variable or multiple variables.
"""

temp = data_1[["Age","Years of experience"]].dropna()
temp_2 = data_1[["Age","Current Salary"]].dropna()

""" Creating plot to check the correlation """

# plot to check the correlation
sns.scatterplot(data=temp,x = 'Age', y = 'Years of experience')

# checking pearson correlation
corr,_ = pearsonr(temp["Age"],temp["Years of experience"])

# checking spearmans correlation
coeff, p = spearmanr(temp["Age"],temp["Years of experience"])

""" Spearsman correlation values comes out to be 0.6223 """

""" Performing Stochastic Regression Imputation to fill age values """

# first build the linear regression model 
lr = LinearRegression()
temp_x = temp["Years of experience"].values
temp_y = temp["Age"].values
temp_x = temp_x.reshape(-1,1)
temp_y = temp_y.reshape(-1,1)

lr.fit(temp_x,temp_y)

# data to fill in age column where there is some value for the years of experience
# this dataframe has all the values of years of experience but not the values of age
temp_3 = data_1[data_1['Age'].isnull() & data_1['Years of experience'].notnull()][["Age","Years of experience"]]

temp_to_predict = temp_3["Years of experience"].values
temp_to_predict = temp_to_predict.reshape(-1,1)

predicted_age = lr.predict(temp_to_predict)

# to get the stochastic part of the values, we have to calculate the residual, variance and noise vector fromt the distribution

residuals = temp_y - lr.predict(temp_x) 

varaince = residuals.var()

noise = np.random.normal(0, np.sqrt(varaince), len(predicted_age))

noise_2 = np.array([noise])

noise_2 = np.transpose(noise_2)

predicted_age_stochastic = predicted_age + noise_2

predicted_age_stochastic = pd.DataFrame(predicted_age_stochastic)

# adding the above evaluated values in the empty values in the table

temp_3_copy = temp_3.copy()

predicted_age_stochastic.index = temp_3_copy.index

temp_4 = pd.concat([temp_3_copy, predicted_age_stochastic], join="inner")

# joing the tables
temp_4 = temp_3_copy.join(predicted_age_stochastic)

temp_4 = temp_4.rename(columns={0:'predicted_age'})

temp_4['predicted_age'] = temp_4['predicted_age'].apply(lambda x: int(x))

del temp_4["Age"]

temp_4 = temp_4.rename(columns = {'predicted_age':'Age'})

# filling the values of age in the main dataframe

for index,val in data_1.iterrows():
    if str(val["Age"]) == 'nan' and str(val["Years of experience"]) != 'nan':
        data_1.at[index,'Age'] = temp_4["Age"][index]

sns.heatmap(data_1.isnull(),cbar=False)

""" Bivariate analysis of Years of experience and salary """

sns.scatterplot(data = data_1, x="Years of experience", y="Current Salary")

# checking how strong the relationship is by correlation

temp_5 = data_1[["Years of experience","Current Salary"]].dropna()

corr,_ = pearsonr(temp_5["Years of experience"],temp_5["Current Salary"])

coeff, p = spearmanr(temp_5["Years of experience"],temp_5["Current Salary"])

""" Since the correlation coefficient is not very strong we are ignoring it """

del data_1['Salary one year ago']
del data_1['Salary two years ago'] 

sns.heatmap(data_1.isnull(),cbar=False)

data_1_drop = data_1.dropna(how="any").reset_index(drop=True)

""" Detecting outliers in the data """

from sklearn.ensemble import IsolationForest
from matplotlib.colors import ListedColormap

temp_6 = data_1_drop[['Age','Years of experience','Current Salary']]

clf = IsolationForest(random_state=0).fit(temp_6)

temp_7 = clf.predict(temp_6)

temp_7_outlier = pd.DataFrame(temp_7)

temp_7_outlier = temp_7_outlier.rename(columns = {0:'outliers'})

temp_6 = pd.DataFrame(temp_6)

temp_6 = temp_6.join(temp_7_outlier) 

# visulaizing outliers
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x = temp_6['Age']
y = temp_6['Years of experience']
z = temp_6['Current Salary']
w = temp_6['outliers']

ax.set_xlabel("Age")
ax.set_ylabel("Experience")
ax.set_zlabel("Current Salary")

sc = ax.scatter(x, y, z, c = w)
plt.show()

# checking how many outliers are there in the data
temp_6.groupby('outliers')['outliers'].count()

""" Visulaizing each variable one by one now """

# gender and salary
sns.barplot(data = data_1_drop, x = 'Gender', y = 'Current Salary' )

# check if the difference between the gender wise slaray distribution is significan tor not using ANOVA test

from scipy.stats import f_oneway

temp_8 = data_1_drop[['Gender','Current Salary']]
temp_9 = temp_8[['Gender','Current Salary']].groupby(['Gender'])

f_oneway(temp_9.get_group('M')["Current Salary"],temp_9.get_group('F')["Current Salary"])

# since the values F one way statistic comes out to be significant with very small P - value is is confirmed that the salary depends on the gender and variance between these two categories are considerabley high
sns.boxplot(data=data_1_drop, x='Your level',y='Age')

# check if the age wise seniority is significant or not
temp_10 = data_1_drop[['Age','Your level']]
temp_11 = temp_10[['Age','Your level']].groupby(['Your level'])

f_oneway(temp_11.get_group('Middle')["Age"],temp_11.get_group('Junior')["Age"])

# salary diff between senior-junior, senior-middle are significant but not between junior and middle using ANOVA tests.

# check if the city and current salart is significant or not
temp_12 = data_1_drop[['City','Current Salary']]
temp_12 = temp_12.where(temp_12['City'] == 'Berlin' | temp_12["City"] == 'München')

temp_14 = temp_12.query("City == 'München' or City == 'Berlin'")

temp_13 = temp_14.groupby('City')['City'].count()

temp_15 = temp_14[['City','Current Salary']].groupby(['City'])

sns.boxplot(data=temp_14, x='City',y='Current Salary')

f_oneway(temp_15.get_group('München')["Current Salary"],temp_15.get_group('Berlin')["Current Salary"])

# Salary does not depend on the city where they are living in currently
