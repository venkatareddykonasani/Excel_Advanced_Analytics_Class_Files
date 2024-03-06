# Import Give me some Credit\cs-training.csv
import pandas as pd
loans=pd.read_csv("D:\\Google Drive\\Training\\Datasets\\Give me some Credit\\cs-training.csv")
loans

#What are number of rows and columns
loans.shape

#Are there any suspicious variables?
loans.columns.values

#Display the variable formats
loans.dtypes

#Print the first 10 observations
loans.head(10)

#Do we have any unique identifier?
loans.columns.values

# # Lab: Frequencies
#What are the categorical and discrete variables? What are the continues variables.
loans.dtypes
loans.head()

#Find the frequencies of all class variables in the data 
loans.columns.values

#Find the frequencies of all class variables in the data 
loans['SeriousDlqin2yrs'].value_counts()
loans['age'].value_counts(sort=False)
loans['NumberOfTime30-59DaysPastDueNotWorse'].value_counts(sort=False)
loans['NumberOfOpenCreditLinesAndLoans'].value_counts(sort=False)
loans['NumberOfTimes90DaysLate'].value_counts(sort=False)
loans['NumberRealEstateLoansOrLines'].value_counts(sort=False)
loans['NumberOfTime60-89DaysPastDueNotWorse'].value_counts(sort=False)
loans['NumberOfDependents'].value_counts(sort=False)



#Are there any   variables with missing values?
loans.isnull().sum()

# # LAB: Continuous variables summary


#List down the continuous variables
loans.dtypes
loans.head()

#Find summary statistics for each variable. Min, Max, Median, Mean, sd, Var
loans['RevolvingUtilizationOfUnsecuredLines'].describe()
loans['MonthlyIncome'].describe()

import numpy as np
variance = np.var(loans['RevolvingUtilizationOfUnsecuredLines'])
variance        

import numpy as np
np.var(loans['MonthlyIncome'])
np.std(loans['RevolvingUtilizationOfUnsecuredLines'])
np.std(loans['MonthlyIncome'])

#Find Quartiles for each of the variables
loans['RevolvingUtilizationOfUnsecuredLines'].describe()
loans['MonthlyIncome'].describe()

#Create Box plots and identify outliers
import matplotlib.pyplot as plt
loans.boxplot(column="RevolvingUtilizationOfUnsecuredLines")
loans.boxplot(column="MonthlyIncome")


#Find the percentage of missing values
loans['MonthlyIncome'].isnull().sum()
loans['MonthlyIncome'].isnull().sum()/len(loans)

#Find Percentiles and find percentage of outliers, if any P1, p5,p10,q1(p25),q3(p75), p90,p99 
util_percentiles=loans['RevolvingUtilizationOfUnsecuredLines'].quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.80, 0.9,0.91,0.95,0.96,0.97,0.975,0.98,0.99,1])
round(util_percentiles,2)

# # LAB: Data Cleaning Scenario-1
#What percent are missing values in RevolvingUtilizationOfUnsecuredLines?
#Get the detailed percentile distribution
util_percentiles=loans['RevolvingUtilizationOfUnsecuredLines'].quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.80, 0.9,0.91,0.95,0.96,0.97,0.975,0.98,0.99,1])
round(util_percentiles,2)

#Clean the variable, and create a new variable by removing all the issues
#If utilization is more than 1 then it can be replaced by median
median_util=loans['RevolvingUtilizationOfUnsecuredLines'].median()
median_util


util_temp_bool_vect=loans['RevolvingUtilizationOfUnsecuredLines']>1
util_temp_bool_vect.value_counts()

loans['util_new']=loans['RevolvingUtilizationOfUnsecuredLines']
loans['util_new'][util_temp_bool_vect]=median_util 
loans['util_new']

# percentile distribution for new variable
util_percentiles1=loans['util_new'].quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.80, 0.9,0.91,0.95,0.96,0.97,0.975,0.98,0.99,1])
round(util_percentiles1,2)

# # LAB: Data Cleaning Scenario-2
#What is the issue with NumberOfTime30_59DaysPastDueNotW
#Draw a frequency table
freq_table_30dpd=loans['NumberOfTime30-59DaysPastDueNotWorse'].value_counts(sort=False)
freq_table_30dpd

#One month defaults frequency can't be beyond 24 in last 24 months
#What percent of the values are erroneous?
freq_table_30dpd[13:len(freq_table_30dpd)]
freq_table_30dpd[13:len(freq_table_30dpd)].sum()/freq_table_30dpd.sum()

#Clean the variable- Look at the cross tab of variable vs target. Impute based on target .
#Cross tab with target
import pandas as pd
cross_tab_30dpd_target=pd.crosstab(loans['NumberOfTime30-59DaysPastDueNotWorse'],loans['SeriousDlqin2yrs'])
cross_tab_30dpd_target

#Cross tab row Percentages
cross_tab_30dpd_target_percent=cross_tab_30dpd_target.astype(float).div(cross_tab_30dpd_target.sum(axis=1), axis=0)
round(cross_tab_30dpd_target_percent,2)

#Percentage of 0 and 1 are of 98 is near to percentages of 6. 
#Replacing error values with 6

loans['num_30_59_dpd_new']=loans['NumberOfTime30-59DaysPastDueNotWorse']
loans['num_30_59_dpd_new'][loans['num_30_59_dpd_new']>12]=6
loans['num_30_59_dpd_new']

loans['num_30_59_dpd_new'].value_counts(sort=False)


# # Data Cleaning Scenario-3
#Find the missing value percentage in monthly income
loans['MonthlyIncome'].isnull().sum()
loans['MonthlyIncome'].isnull().sum()/len(loans)
#Once identified where missing values exist, the next task usually is to fill them (data imputation). Depending upon the context,
#in this case, I am assigning median value to all those positions where missing value is present:

loans['MonthlyIncome_ind']=1
loans['MonthlyIncome_ind'][loans['MonthlyIncome'].isnull()]=0
loans['MonthlyIncome_ind'].value_counts(sort=False)



loans['MonthlyIncome_new']=loans['MonthlyIncome']
loans['MonthlyIncome_new'][loans['MonthlyIncome'].isnull()]=loans['MonthlyIncome'].median()
round(loans['MonthlyIncome_new'].describe())