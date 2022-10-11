#!/usr/bin/env python
# coding: utf-8

# In[114]:


import os
import math
import pandas as pd
import numpy as np
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pylab as plt
import wquantiles
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV as cv
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier,NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler 
import plotly.express as px

style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)
##
dataSet = pd.read_csv("D:\Spring22\Data Science\Project\students_adaptability_level_online_education.csv")

df = pd.read_csv('D:\Spring22\Data Science\Project\students_adaptability_level_online_education.csv')

df.head()
df.shape


# In[46]:


print(dataSet)
print(dataSet.shape)


# In[78]:


dataSet.dtypes


# In[77]:


from sklearn.preprocessing import LabelEncoder
enCode = dataSet['Gender'].values
dataSet['Gender'] = LabelEncoder().fit_transform(enCode)

enCode = dataSet['Education Level'].values
dataSet['Education Level'] = LabelEncoder().fit_transform(enCode)


enCode = dataSet['Institution Type'].values
dataSet['Institution Type'] = LabelEncoder().fit_transform(enCode)


enCode = dataSet['IT Student'].values
dataSet['IT Student'] = LabelEncoder().fit_transform(enCode)

enCode = dataSet['Location'].values
dataSet['Location'] = LabelEncoder().fit_transform(enCode)

enCode = dataSet['Load-shedding'].values
dataSet['Load-shedding'] = LabelEncoder().fit_transform(enCode)

enCode = dataSet['Financial Condition'].values
dataSet['Financial Condition'] = LabelEncoder().fit_transform(enCode)

enCode = dataSet['Internet Type'].values
dataSet['Internet Type'] = LabelEncoder().fit_transform(enCode)

enCode = dataSet['Network Type'].values
dataSet['Network Type'] = LabelEncoder().fit_transform(enCode)

enCode = dataSet['Self Lms'].values
dataSet['Self Lms'] = LabelEncoder().fit_transform(enCode)

enCode = dataSet['Device'].values
dataSet['Device'] = LabelEncoder().fit_transform(enCode)

enCode = dataSet['Adaptivity Level'].values
dataSet['Adaptivity Level'] = LabelEncoder().fit_transform(enCode)

#Removing 'Age' and 'Class Duration' attribute as it is not of focus.
dataSet.pop('Age')
dataSet.pop('Class Duration')

#Removing null values if exist.
dataSet = dataSet.dropna();

#Display dataframe
dataSet.head()


# In[25]:


#Finding correlation

dataplot = sns.heatmap(dataSet.corr(), cmap="YlGnBu", annot=True)

# displaying heatmap
plt.show()
dataSet.corr()


# In[51]:


from scipy import stats
for i in df.columns.tolist() :
    test = df.groupby([i, 'Adaptivity Level'])['Gender'].count().unstack()
    _, p_val, _, _ = stats.chi2_contingency(test)
    if p_val.round(5) <=0.5:
        print(i, '- Adaptivity Level p-value:', p_val )


# In[52]:


#SINCE ATTRIBUTES HAS LOW P-VALUES, HENCE DATA IS STATISTICALLY SIGNIFICANT, WE CAN PROCEED FURTHER FOR TRAINING


# In[67]:


#DROP THE MEASURING ATTRIBUTE
X = df.drop('Adaptivity Level', axis=1)
y = df['Adaptivity Level']

#TAKE TRAINING SAMPLE OF SIZE 10 (RANDOM SAMPLING)
rus = RandomOverSampler(random_state=10)
X_rus, y_rus = rus.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus, stratify=y_rus, test_size= 0.3, random_state=0)

print('X_train shape:''\n' 'Rows:', X_train.shape[0],'\n' 'Columns:', X_train.shape[1])

ohe = OneHotEncoder(drop='first', sparse=False)
X_train = ohe.fit_transform(X_train)
X_test = ohe.transform(X_test)


# In[4]:


boxplot = dataSet.boxplot(column = ['Education Level', 'IT Student', 'Institution Type', 'Internet Type', 'Adaptivity Level', 'Device'])
print(boxplot)


# In[67]:


dataSet.hist(figsize=(10,10),bins=5)


# In[68]:


ax = sns.scatterplot(x="Gender", y="Adaptivity Level", data=dataSet)
ax.set_title("Gender vs. Adaptivity Level")


# In[115]:


top_prod = df.groupby('Adaptivity Level').size().reset_index().rename(columns={0: 'total'}).sort_values('total', ascending=False).head(5)
fig = px.pie(top_prod, values='total', names='Adaptivity Level', color_discrete_sequence=px.colors.sequential.thermal, title="Students' adaptivity level")
fig.show()


# In[118]:


fig = px.parallel_categories(df,dimensions=['Gender', 'Institution Type' , 'Adaptivity Level'],
                 color_continuous_scale=px.colors.sequential.Inferno,
    )
fig.show()


# In[79]:


# We can use the `LinearRegression` model from _scikit-learn_.
predictors = ['Gender']
outcome = 'IT Student'

model = LinearRegression()
model.fit(dataSet[predictors], dataSet[outcome])

print(f'Intercept: {model.intercept_:.3f}')
print(f'Coefficient Exposure: {model.coef_[0]:.3f}')

fig, ax = plt.subplots(figsize=(4, 4))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('IT Student')
ax.set_ylabel('Gender')
ax.plot((0, 23), model.predict([[0], [23]]))
ax.text(0.4, model.intercept_, r'$b_0$', size='larger')


# In[80]:



subset = ['Gender', 'Education Level', 'Institution Type', 'IT Student', 
          'Internet Type', 'Device']

predictors = ['Education Level', 'Institution Type', 'IT Student', 'Internet Type', 'Device']
outcome = 'Adaptivity Level'

dataSet_lm = LinearRegression()
dataSet_lm.fit(dataSet[predictors], dataSet[outcome])

print(f'Intercept: {dataSet_lm.intercept_:.3f}')
print('Coefficients:')
for name, coef in zip(predictors, dataSet_lm.coef_):
    print(f' {name}: {coef}')
#%% Assessing the Model
# _Scikit-learn_ provides a number of metrics to determine the quality of a model. 
# Here we use the `r2_score`.
fitted = dataSet_lm.predict(dataSet[predictors])
RMSE = np.sqrt(mean_squared_error(dataSet[outcome], fitted))
r2 = r2_score(dataSet[outcome], fitted)
print(f'RMSE: {RMSE:.0f}')
print(f'r2: {r2:.4f}')


# In[95]:


#FINDING ACCURACY OF INITIAL DATA

y_pred = cv.predict(X_test)
print(classification_report(y_pred, y_test))


# In[96]:


#MODEL CHANGING FOR IMPROVING ACCURACY

svc = SVC()
svc_params = {'kernel':['linear', 'poly', 'rbf']}
cv = GridSearchCV(svc, svc_params, cv=5)
cv.fit(X_train, y_train)


# In[97]:


y_pred = cv.predict(X_test)
print(classification_report(y_pred, y_test))


# In[98]:


#USING KNN CLASSIFIER TO IMPROVE ACCURACY

knc = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
y_pred = knc.predict(X_test)
print(classification_report(y_pred, y_test))


# In[ ]:


#SEEMS TO BE NOT A GOOD MODEL FOR KNN CLASSIFIER, CHANGING OUR MODEL


# In[100]:


knc = KNeighborsClassifier(n_neighbors=3)
nca = NeighborhoodComponentsAnalysis(random_state=10)
pipeline = Pipeline([
    ('nca', nca), 
    ('knc', knc)
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_pred, y_test))


# In[ ]:


#THE MODEL IMPROVED BUT REMAINS AT 0.89 ACCURACY


# In[101]:


bagg = BaggingClassifier(base_estimator= knc)
params = {'max_features':np.linspace(0.1, 1, 10), 'max_samples':np.linspace(0.1, 0.2, 10)}
cv = GridSearchCV(bagg, params, cv=5)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print(classification_report(y_pred, y_test))


# In[106]:


hgbc = HistGradientBoostingClassifier(loss='categorical_crossentropy')
hgbc.fit(X_train, y_train)
y_pred = hgbc.predict(X_test)
print(classification_report(y_pred, y_test))


# In[ ]:


#MODEL FITTING PROCESS


# In[107]:


AB = AdaBoostClassifier(n_estimators=30).fit(X_train,y_train)
print('Train data:')
print(classification_report(AB.predict(X_train),y_train))
print('Test data:')
print(classification_report(AB.predict(X_test),y_test))


# In[111]:


lr = LogisticRegression(solver='newton-cg').fit(X_train,y_train)
print('train data report')
print(classification_report(lr.predict(X_train),y_train))
print('test data report')
print(classification_report(lr.predict(X_test),y_test))


# In[108]:


KNN = KNeighborsClassifier(n_neighbors=40,weights="distance").fit(X_train,y_train)
print('Train')
print(classification_report(KNN.predict(X_train),y_train))
print('Test')
print(classification_report(KNN.predict(X_test),y_test))


# In[ ]:




