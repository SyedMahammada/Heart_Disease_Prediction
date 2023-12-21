#!/usr/bin/env python
# coding: utf-8

# # 1.Introduction

# ### Problem Statement
# 
# We have just been hired as a Data Scientist at a Hospital with an alarming number of patients coming in reporting various cardiac symptoms. A cardiologist measures vitals & hands you this data to perform Data Analysis and predict whether certain patients have Heart Disease. 
#    
# 

# ### Goal
# 
# * Predict whether a patient should be diagnosed with Heart Disease. This is a binary outcome.
#     > Positive (+) = 1, patient diagnosed with Heart Disease.
#     
#     > Negative (-) = 0, patient not diagnosed with Heart Disease.
# * Experiment with various Classification Models & see which yields greatest accuracy.
# * Examine trends & correlations within our data.
# * Determine which features are most important to Positive/Negative Heart Disease diagnosis.

# ### Features & Predictor:
# 
# 
# Our Predictor (Y, Positive or Negative diagnosis of Heart Disease) is determined by 13 features (X):
# 
# 1. age (#)
# 2. sex : 1= Male, 0= Female (Binary)
# 3. (cp)chest pain type (4 values -Ordinal):Value 1: typical angina ,Value 2: atypical angina, Value 3: non-anginal pain , Value 4: asymptomatic
# 4. (trestbps) resting blood pressure (#)
# 5. (chol) serum cholesterol in mg/dl (#)
# 6. (fbs)fasting blood sugar > 120 mg/dl(Binary)(1 = true; 0 = false)
# 7. (restecg) resting electrocardiography results(values 0,1,2)
# 8. (thalach) maximum heart rate achieved (#)
# 9. (exang) exercise induced angina (binary) (1 = yes; 0 = no)
# 10. (oldpeak) = ST depression induced by exercise relative to rest (#)
# 11. (slope) of the peak exercise ST segment (Ordinal) (Value 1: up sloping , Value 2: flat , Value 3: down sloping )
# 12. (ca) number of major vessels (0–3, Ordinal) colored by fluoroscopy
# 13. (thal) maximum heart rate achieved — (Ordinal): 3 = normal; 6 = fixed defect; 7 = reversible defect
# 
# Note: Our data has 3 types of data:
# 
# Continuous (#): which is quantitative data that can be measured
# 
# Ordinal Data: Categorical data that has a order to it (0,1,2,3, etc)
# 
# Binary Data: data whose unit can take on only two possible states ( 0 &1 )
# 
# 

# # 2.Data Wrangling

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("heart.csv")
data.head(10)


# In[3]:


print("(Rows,Columns) : " +str(data.shape))
data.columns 


# In[4]:


data.nunique(axis=0)# returns the number of unique values for each variable.


# In[5]:


#summarizes the count, mean, standard deviation, min, and max for numeric variables.
data.describe()


# In[6]:


# Display the Missing Values

print(data.isna().sum())


# In[7]:


data['target'].value_counts()


# # 3.Exploratory Data Analysis

# ### Correlations
# 
# Correlation Matrix- let’s you see correlations between all variables.
# 
# Within seconds, you can see whether something is positively or negatively correlated with our predictor (target).

# In[8]:


# calculate correlation matrix

corr = data.corr()
plt.subplots(figsize=(15,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, 
            cmap=sns.diverging_palette(220, 20, as_cmap=True))
sns.heatmap(corr, xticklabels=corr.columns,
            yticklabels=corr.columns, 
            annot=True,
            cmap=sns.diverging_palette(220, 20, as_cmap=True))


# In[9]:


subData = data[['age','trestbps','chol','thalach','oldpeak']]
sns.pairplot(subData)


# In[10]:


sns.catplot(x="target", y="oldpeak", hue="slope", kind="bar", data=data);

plt.title('ST depression (induced by exercise relative to rest) vs. Heart Disease',size=25)
plt.xlabel('Heart Disease',size=20)
plt.ylabel('ST depression',size=20)


# 
# 
# ST segment depression occurs because when the ventricle is at rest and therefore repolarized. If the trace in the ST segment is abnormally low below the baseline, this can lead to this Heart Disease. This is supports the plot above because low ST Depression yields people at greater risk for heart disease. While a high ST depression is considered normal & healthy. The “slope” hue, refers to the peak exercise ST segment, with values: 0: upsloping , 1: flat , 2: downsloping). Both positive & negative heart disease patients exhibit equal distributions of the 3 slope categories.
# 
# 

# ## Violin & Box Plots

# The advantages of showing the Box & Violin plots is that it shows the basic statistics of the data, as well as its distribution. These plots are often used to compare the distribution of a given variable across some categories.

# In[11]:


plt.figure(figsize=(12,8))
sns.violinplot(x= 'target', y= 'oldpeak',hue="sex", inner='quartile',data= data )
plt.title("Thalach Level vs. Heart Disease",fontsize=20)
plt.xlabel("Heart Disease Target", fontsize=16)
plt.ylabel("Thalach Level", fontsize=16)



# We can see that the overall shape & distribution for negative & positive patients differ vastly. Positive patients exhibit a lower median for ST depression level & thus a great distribution of their data is between 0 & 2, while negative patients are between 1 & 3. In addition, we don’t see many differences between male & female target outcomes.
# 

# In[12]:


plt.figure(figsize=(12,8))
sns.boxplot(x= 'target', y= 'thalach',hue="sex", data=data )
plt.title("ST depression Level vs. Heart Disease", fontsize=20)
plt.xlabel("Heart Disease Target",fontsize=16)
plt.ylabel("ST depression induced by exercise relative to rest", fontsize=16)


# Positive patients exhibit a heightened median for ST depression level, while negative patients have lower levels. In addition, we don’t see many differences between male & female target outcomes, expect for the fact that males have slightly larger ranges of ST Depression.

# ## Filtering data by positive & negative Heart Disease patient

# In[13]:


# Filtering data by POSITIVE Heart Disease patient
pos_data = data[data['target']==1]
pos_data.describe()


# In[14]:


# Filtering data by NEGATIVE Heart Disease patient
neg_data = data[data['target']==0]
neg_data.describe()


# In[15]:


print("(Positive Patients ST depression): " + str(pos_data['oldpeak'].mean()))
print("(Negative Patients ST depression): " + str(neg_data['oldpeak'].mean()))


# In[16]:


print("(Positive Patients thalach): " + str(pos_data['thalach'].mean()))
print("(Negative Patients thalach): " + str(neg_data['thalach'].mean()))


# From comparing positive and negative patients we can see there are vast differences in means for many of our 13 Features. From examining the details, we can observe that positive patients experience heightened maximum heart rate achieved (thalach) average. In addition, positive patients exhibit about 1/3rd the amount of ST depression induced by exercise relative to rest (oldpeak).
# 
# 

# # 4.Machine Learning + Predictive Analytics

# ## Prepare Data for Modeling

# Assign the 13 features to X, & the last column to our classification predictor, y

# In[17]:


X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# Split: the data set into the Training set and Test set

# In[18]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)


# Normalize: Standardizing the data will transform the data so that its distribution will have a mean of 0 and a standard deviation of 1.

# In[19]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# ## Modeling /Training

# Now we’ll Train various Classification Models on the Training set & see which yields the highest accuracy. We will compare the accuracy of Logistic Regression, K-NN (k-Nearest Neighbours), SVM (Support Vector Machine), Naives Bayes Classifier, Decision Trees, Random Forest.
# 
# Note: these are all supervised learning models.
# 

# ### Model 1: Logistic Regression

# In[20]:


from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(random_state=1) # get instance of model
model1.fit(x_train, y_train) # Train/Fit model 

y_pred1 = model1.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred1)) # output accuracy


# ### Model 2: K-NN (K-Nearest Neighbors)

# In[21]:


from sklearn.metrics import classification_report 
from sklearn.neighbors import KNeighborsClassifier

model2 = KNeighborsClassifier() # get instance of model
model2.fit(x_train, y_train) # Train/Fit model 

y_pred2 = model2.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred2)) # output accuracy


# ### Model 3: SVM (Support Vector Machine)

# In[22]:


from sklearn.metrics import classification_report 
from sklearn.svm import SVC

model3 = SVC(random_state=1) # get instance of model
model3.fit(x_train, y_train) # Train/Fit model 

y_pred3 = model3.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred3)) # output accuracy


# ### Model 4: Naives Bayes Classifier

# In[23]:


from sklearn.metrics import classification_report 
from sklearn.naive_bayes import GaussianNB

model4 = GaussianNB() # get instance of model
model4.fit(x_train, y_train) # Train/Fit model 

y_pred4 = model4.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred4)) # output accuracy


# ### Model 5: Decision Trees

# In[24]:


from sklearn.metrics import classification_report 
from sklearn.tree import DecisionTreeClassifier

model5 = DecisionTreeClassifier(random_state=1) # get instance of model
model5.fit(x_train, y_train) # Train/Fit model 

y_pred5 = model5.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred5)) # output accuracy


# ### Model 6: Random Forest 🏆

# In[25]:


from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier

model6 = RandomForestClassifier(random_state=1)# get instance of model
model6.fit(x_train, y_train) # Train/Fit model 

y_pred6 = model6.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred6)) # output accuracy


# #### From comparing the 6 models, we can conclude that Model 6: Random Forest yields the highest accuracy. With an accuracy of 80%. 🏆

# Precision, Recall, F1-score and Support:
# 
# Precision : be “how many are correctly classified among that class”
# 
# Recall : “how many of this class you find over the whole number of element of this class”
# 
# F1-score : harmonic mean of precision and recall values.
# F1 score reaches its best value at 1 and worst value at 0.
# F1 Score = 2 x ((precision x recall) / (precision + recall))
# 
# Support: # of samples of the true response that lie in that class.

# ## Making the Confusion Matrix

# 21 is the amount of True Positives in our data, while 28 is the amount of True Negatives.
# 
# 9 & 3 are the number of errors.
# 
# There are 9 Type 1 errors (False Positives)- You predicted positive and it’s false.
# 
# There are 3 Type 2 errors (False Negatives)- You predicted negative and it’s false.
# 
# Hence, if we calculate the accuracy its # Correct Predicted/ # Total.
# In other words, where TP, FN, FP and TN represent the number of true positives, false negatives, false positives and true negatives.
# 
# Accuracy = (TP + TN)/(TP + TN + FP + FN).
# Accuracy =(21+28)/(21+28+9+3) = 0.80 = 80% accuracy
# 
# 

# ## Feature Importance

# Feature Importance provides a score that indicates how helpful each feature was in our model.
# 
# The higher the Feature Score, the more that feature is used to make key decisions & thus the more important it is.

# In[26]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred6)
print(cm)
accuracy_score(y_test, y_pred6)


# ### How To Interpret Confusion Matrix:

# In[27]:


# get importance
importance = model6.feature_importances_

# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))


# In[28]:


index= data.columns[:-1]
importance = pd.Series(model6.feature_importances_, index=index)
importance.nlargest(13).plot(kind='barh', colormap='winter')


# From the Feature Importance graph above, we can conclude that the top 4 significant features were chest pain type (cp), maximum heart rate achieved (thalach), number of major vessels (ca), and ST depression induced by exercise relative to rest (oldpeak).
# 

# # Predictions

# ### Scenario: 
#     A patient develops cardiac symptoms & you input his vitals into the Machine Learning Algorithm.

# 
# He is a 20 year old male, with a chest pain value of 2 (atypical angina), with resting blood pressure of 110.
# 
# In addition he has a serum cholestoral of 230 mg/dl.
# 
# He is fasting blood sugar > 120 mg/dl.
# 
# He has a resting electrocardiographic result of 1.
# 
# The patients maximum heart rate achieved is 140.
# 
# Also, he was exercise induced angina.
# 
# His ST depression induced by exercise relative to rest value was 2.2.
# 
# The slope of the peak exercise ST segment is flat.
# 
# He has no major vessels colored by fluoroscopy, and in addition his maximum heart rate achieved is a reversible defect.
# 
# Based on this information, can you classify this patient with Heart Disease?

# In[29]:


print(model6.predict(sc.transform([[20,1,2,110,230,1,1,140,1,2.2,2,0,2]])))


# * Outputs Binary 1 -> meaning Positive Diagnosis of Heart Disease

# #### Predicting the Test set results:

# First value represents our predicted value, Second value represents our actual value.
# 
# If the values match, then we predicted correctly.

# In[30]:


y_pred = model6.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# We can see that our results are very accurate (80%) !

# ## Conclusions

# ### 1. Out of the 13 features we examined, the top 4 significant features that helped us classify between a positive & negative Diagnosis were chest pain type (cp), maximum heart rate achieved (thalach), number of major vessels (ca), and ST depression induced by exercise relative to rest (oldpeak).
# 
# ### 2. Our machine learning algorithm can now classify patients with Heart Disease. Now we can properly diagnose patients, & get them the help they needs to recover. By diagnosing detecting these features early, we may prevent worse symptoms from arising later.
# 
# ### 3. Our Random Forest algorithm yields the highest accuracy, 80%. Any accuracy above 70% is considered good, but be careful because if your accuracy is extremely high, it may be too good to be true (an example of Over fitting). Thus, 80% is the ideal accuracy!
