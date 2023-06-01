#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# ## Data Exploring

# In[2]:


df=pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# ## Pre-Processing

# In[5]:


df=df.drop_duplicates()
df.head()


# In[6]:


df.info()


# In[7]:


unique_values = {}
for col in df.columns:
    unique_values[col] = df[col].value_counts().shape[0]

pd.DataFrame(unique_values, index=['unique value count']).transpose()


# In[8]:


def v_counts(dataframe):
    for i in dataframe :
        print(dataframe[i].value_counts())
        print("_____________________________________________________________________________")


# In[9]:


v_counts(df)


# In[10]:


#df=df.drop(["Income","Education","AnyHealthcare","NoDocbcCost","CholCheck"],axis=1)


# In[11]:


df.describe()


# In[12]:


df.isnull().sum()


# ## converting data to catgorical to better understanding and EDA

# In[13]:


df["BMICAT"]= df.apply(lambda x:"under weight" if x["BMI"]<18.5 else x["BMI"],axis=1)
df["BMICAT"]= df.apply(lambda x:"Noraml weight" if 18.5<=x["BMI"]<25.0 else x["BMICAT"],axis=1)
df["BMICAT"]= df.apply(lambda x:"over weight" if 25.0<=x["BMI"]<30.0 else x["BMICAT"],axis=1)
df["BMICAT"]= df.apply(lambda x:"obese" if x["BMI"]>=30.0 else x["BMICAT"],axis=1)


# In[14]:


bmicat = {'under weight':0, 'Noraml weight':1, 'over weight':2, 'obese':3}
df['BMICAT'] = df['BMICAT'].replace(bmicat)


# In[15]:


df=df.drop("BMI",axis=1)


# In[16]:


df.Diabetes_binary.value_counts()


# In[17]:


plt.bar(x=["No Diabetes","Diabetes"],height=df.Diabetes_binary.value_counts())

plt.show()


# Dataset is balanced

# # EDA

# In[18]:



plt.figure(figsize = (15,10))
sns.heatmap(df.corr(),annot=True , cmap ='YlOrRd' )
plt.title("correlation of feature")


# In[19]:



df.hist(figsize=(20,15));


# In[20]:


cols = ['HighBP', 'HighChol', 'CholCheck','Smoker',
       'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk']


# In[21]:


def create_plot_pivot(data2, x_column):
    """ Create a pivot table for satisfaction versus another rating for easy plotting. """
    _df_plot = data2.groupby([x_column, 'Diabetes_binary']).size()     .reset_index().pivot(columns='Diabetes_binary', index=x_column, values=0)
    return _df_plot


# In[22]:


data2 = df.copy() 


# In[23]:


# That help us to show the relation between features clearly

data2.Age[data2['Age'] == 1] = '18 to 24'
data2.Age[data2['Age'] == 2] = '25 to 29'
data2.Age[data2['Age'] == 3] = '30 to 34'
data2.Age[data2['Age'] == 4] = '35 to 39'
data2.Age[data2['Age'] == 5] = '40 to 44'
data2.Age[data2['Age'] == 6] = '45 to 49'
data2.Age[data2['Age'] == 7] = '50 to 54'
data2.Age[data2['Age'] == 8] = '55 to 59'
data2.Age[data2['Age'] == 9] = '60 to 64'
data2.Age[data2['Age'] == 10] = '65 to 69'
data2.Age[data2['Age'] == 11] = '70 to 74'
data2.Age[data2['Age'] == 12] = '75 to 79'
data2.Age[data2['Age'] == 13] = '80 or older'

data2.Diabetes_binary[data2['Diabetes_binary'] == 0] = 'No Diabetes'
data2.Diabetes_binary[data2['Diabetes_binary'] == 1] = 'Diabetes'

data2.HighBP[data2['HighBP'] == 0] = 'No High'
data2.HighBP[data2['HighBP'] == 1] = 'High BP'

data2.HighChol[data2['HighChol'] == 0] = 'No High Cholesterol'
data2.HighChol[data2['HighChol'] == 1] = 'High Cholesterol'

data2.CholCheck[data2['CholCheck'] == 0] = 'No Cholesterol Check in 5 Years'
data2.CholCheck[data2['CholCheck'] == 1] = 'Cholesterol Check in 5 Years'

data2.Smoker[data2['Smoker'] == 0] = 'No'
data2.Smoker[data2['Smoker'] == 1] = 'Yes'

data2.Stroke[data2['Stroke'] == 0] = 'No'
data2.Stroke[data2['Stroke'] == 1] = 'Yes'

data2.HeartDiseaseorAttack[data2['HeartDiseaseorAttack'] == 0] = 'No'
data2.HeartDiseaseorAttack[data2['HeartDiseaseorAttack'] == 1] = 'Yes'

data2.PhysActivity[data2['PhysActivity'] == 0] = 'No'
data2.PhysActivity[data2['PhysActivity'] == 1] = 'Yes'

data2.Fruits[data2['Fruits'] == 0] = 'No'
data2.Fruits[data2['Fruits'] == 1] = 'Yes'

data2.Veggies[data2['Veggies'] == 0] = 'No'
data2.Veggies[data2['Veggies'] == 1] = 'Yes'

data2.HvyAlcoholConsump[data2['HvyAlcoholConsump'] == 0] = 'No'
data2.HvyAlcoholConsump[data2['HvyAlcoholConsump'] == 1] = 'Yes'

data2.AnyHealthcare[data2['AnyHealthcare'] == 0] = 'No'
data2.AnyHealthcare[data2['AnyHealthcare'] == 1] = 'Yes'

data2.NoDocbcCost[data2['NoDocbcCost'] == 0] = 'No'
data2.NoDocbcCost[data2['NoDocbcCost'] == 1] = 'Yes'

data2.GenHlth[data2['GenHlth'] == 5] = 'Excellent'
data2.GenHlth[data2['GenHlth'] == 4] = 'Very Good'
data2.GenHlth[data2['GenHlth'] == 3] = 'Good'
data2.GenHlth[data2['GenHlth'] == 2] = 'Fair'
data2.GenHlth[data2['GenHlth'] == 1] = 'Poor'

data2.DiffWalk[data2['DiffWalk'] == 0] = 'No'
data2.DiffWalk[data2['DiffWalk'] == 1] = 'Yes'

data2.Sex[data2['Sex'] == 0] = 'Female'
data2.Sex[data2['Sex'] == 1] = 'Male'

data2.Education[data2['Education'] == 1] = 'Never Attended School'
data2.Education[data2['Education'] == 2] = 'Elementary'
data2.Education[data2['Education'] == 3] = 'Junior High School'
data2.Education[data2['Education'] == 4] = 'Senior High School'
data2.Education[data2['Education'] == 5] = 'Undergraduate Degree'
data2.Education[data2['Education'] == 6] = 'Magister'

data2.Income[data2['Income'] == 1] = 'Less Than $10,000'
data2.Income[data2['Income'] == 2] = 'Less Than $10,000'
data2.Income[data2['Income'] == 3] = 'Less Than $10,000'
data2.Income[data2['Income'] == 4] = 'Less Than $10,000'
data2.Income[data2['Income'] == 5] = 'Less Than $35,000'
data2.Income[data2['Income'] == 6] = 'Less Than $35,000'
data2.Income[data2['Income'] == 7] = 'Less Than $35,000'
data2.Income[data2['Income'] == 8] = '$75,000 or More'


# In[24]:


fig, ax = plt.subplots(3, 4, figsize=(20,20))
axe = ax.ravel()

c = len(cols)

for i in range(c):
    create_plot_pivot(data2, cols[i]).plot(kind='bar',stacked=True, ax=axe[i])
    axe[i].set_xlabel(cols[i])
    
fig.show()


# In[25]:


data2["Diabetes_binary"].value_counts()


# In[26]:


#checking the value count of Diabetes_binary_str by using countplot
figure1, plot1 = plt.subplots(1,2,figsize=(10,8))

sns.countplot(data2['Diabetes_binary'],ax=plot1[0])


#checking diabetic and non diabetic pepoles average by pie 

labels=["non-Diabetic","Diabetic"]

plt.pie(data2["Diabetes_binary"].value_counts() , labels =labels ,autopct='%.02f' );


# In[27]:


plt.figure(figsize=(10,6))


sns.distplot(df.Education[df.Diabetes_binary == 0], color="y", label="No Diabetic" )
sns.distplot(df.Education[df.Diabetes_binary == 1], color="m", label="Diabetic" )
plt.title("Relation b/w Education and Diabetes")

plt.legend()


# In[28]:


pd.crosstab(df.MentHlth,df.Diabetes_binary).plot(kind="bar",figsize=(30,12),color=['#1CA53B', '#FFA500' ])
plt.title('Diabetes Disease Frequency for MentHlth')
plt.xlabel('MentHlth')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.show()


# In[29]:


pd.crosstab(df.PhysHlth,df.Diabetes_binary).plot(kind="bar",figsize=(30,12),color=['Blue', 'Red' ])
plt.title('Diabetes Disease Frequency for PhysHlth')
plt.xlabel('PhysHlth')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.show()


# In[30]:


pd.crosstab(df.GenHlth,df.Diabetes_binary).plot(kind="bar",figsize=(30,12),color=['Purple', 'Green' ])
plt.title('Diabetes Disease Frequency for GenHlth')
plt.xlabel('GenHlth')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.show()


# # Feature Selection

# In[31]:


df.drop('Diabetes_binary', axis=1).corrwith(df.Diabetes_binary).plot(kind='bar', grid=True, figsize=(20, 8)
, title="Correlation with Diabetes_binary",color="Purple");


# In[32]:


X = df.loc[:,df.columns != 'Diabetes_binary']
X.head()


# In[33]:


y = df.Diabetes_binary
y


# In[34]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=65)


# In[35]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[36]:


X_train.describe()


# In[37]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train[["GenHlth","MentHlth","PhysHlth","Age","BMICAT"]] = mms.fit_transform(X_train[["GenHlth","MentHlth","PhysHlth","Age","BMICAT"]])
X_test[["GenHlth","MentHlth","PhysHlth","Age","BMICAT"]] = mms.transform(X_test[["GenHlth","MentHlth","PhysHlth","Age","BMICAT"]])


# In[38]:


X_train.describe()


# In[39]:


X_train.head()


# In[40]:


y_train


# In[41]:


plt.figure(figsize=(18,18))
cor = X_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Greens, fmt='.2f')
plt.show()


# There is no strong Correlatin between independent variables

# ## Model 1 -Decision Tree Classifier

# In[42]:


model_1 = DecisionTreeClassifier(random_state = 45)
model_1.fit(X_train, y_train)

# Calculate model perfomance
predictions = model_1.predict(X_test)
model_1_score = accuracy_score(y_test, predictions)

print('Accuracy score for Decision Tree is', model_1_score)


# In[43]:


class_names = ["non Diabetes","Diabetes"]
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(model_1, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Greens,
                                 normalize=normalize)
    disp.ax_.set_title(title)


# In[44]:


y_pred_train = model_1.predict(X_train)
y_pred_test = model_1.predict(X_test)


# In[45]:


fpr_log,tpr_log,thres_log = roc_curve(y_test, y_pred_test)
log_precision, log_recall, log_thres = precision_recall_curve(y_test, y_pred_test)


fig, ax = plt.subplots(1,2,figsize=(20,6))
ax[0].plot(fpr_log,tpr_log)
ax[0].plot([0, 1], ls="--")
ax[0].plot([0, 0], [1, 0] , c=".7")
ax[0].plot([1, 1] , c=".7")
ax[0].set_ylabel('True Positive Rate')
ax[0].set_xlabel('False Positive Rate')
print(roc_auc_score(y_test, y_pred_test))


ax[1].plot(log_recall,log_precision)
ax[1].plot([0, 1], ls="--")
ax[1].plot([0, 0], [1, 0] , c=".7")
ax[1].plot([1, 1] , c=".7")
ax[1].set_ylabel('Precision')
ax[1].set_xlabel('Recall')
plt.show()


# In[46]:


accuracy_score(y_train, y_pred_train)


# In[47]:


accuracy_score(y_test, y_pred_test)


# #### Model 1 gets Overfitted

# ## Model 2 -RandomForestClassifier

# In[48]:


# Random Forest
model_2 = RandomForestClassifier(random_state = 45)
model_2.fit(X_train, y_train)

# Calculate model perfomance
predictions = model_2.predict(X_test)
model_2_score = accuracy_score(y_test, predictions)

print('Accuracy score for Random Forest is', model_2_score)


# In[49]:


class_names = ["non Diabetes","Diabetes"]
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(model_2, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Greens,
                                 normalize=normalize)
    disp.ax_.set_title(title)


# In[50]:


y_pred_train = model_2.predict(X_train)
y_pred_test = model_2.predict(X_test)


# In[51]:


fpr_log,tpr_log,thres_log = roc_curve(y_test, y_pred_test)
log_precision, log_recall, log_thres = precision_recall_curve(y_test, y_pred_test)


fig, ax = plt.subplots(1,2,figsize=(20,6))
ax[0].plot(fpr_log,tpr_log)
ax[0].plot([0, 1], ls="--")
ax[0].plot([0, 0], [1, 0] , c=".7")
ax[0].plot([1, 1] , c=".7")
ax[0].set_ylabel('True Positive Rate')
ax[0].set_xlabel('False Positive Rate')
print(roc_auc_score(y_test, y_pred_test))


ax[1].plot(log_recall,log_precision)
ax[1].plot([0, 1], ls="--")
ax[1].plot([0, 0], [1, 0] , c=".7")
ax[1].plot([1, 1] , c=".7")
ax[1].set_ylabel('Precision')
ax[1].set_xlabel('Recall')
plt.show()


# In[52]:


accuracy_score(y_train, y_pred_train)


# In[53]:


accuracy_score(y_test, y_pred_test)


# #### Model 2 gets Overfitted

# ## Model 3 - KNeighborsClassifier

# In[54]:


model_3 = KNeighborsClassifier()
model_3.fit(X_train, y_train)

# Calculate model perfomance
predictions = model_3.predict(X_test)
model_3_score = accuracy_score(y_test, predictions)

print('Accuracy score for KNeighbors is', model_3_score)


# In[55]:


class_names = ["non Diabetes","Diabetes"]
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(model_3, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Greens,
                                 normalize=normalize)
    disp.ax_.set_title(title)


# In[56]:


y_pred_train = model_3.predict(X_train)
y_pred_test = model_3.predict(X_test)


# In[57]:


fpr_log,tpr_log,thres_log = roc_curve(y_test, y_pred_test)
log_precision, log_recall, log_thres = precision_recall_curve(y_test, y_pred_test)


fig, ax = plt.subplots(1,2,figsize=(20,6))
ax[0].plot(fpr_log,tpr_log)
ax[0].plot([0, 1], ls="--")
ax[0].plot([0, 0], [1, 0] , c=".7")
ax[0].plot([1, 1] , c=".7")
ax[0].set_ylabel('True Positive Rate')
ax[0].set_xlabel('False Positive Rate')
print(roc_auc_score(y_test, y_pred_test))


ax[1].plot(log_recall,log_precision)
ax[1].plot([0, 1], ls="--")
ax[1].plot([0, 0], [1, 0] , c=".7")
ax[1].plot([1, 1] , c=".7")
ax[1].set_ylabel('Precision')
ax[1].set_xlabel('Recall')
plt.show()


# In[58]:


accuracy_score(y_train, y_pred_train)


# In[59]:


accuracy_score(y_test, y_pred_test)


# Overfitting

# ## Model 4 - Logistic Regression

# In[60]:


model_4 = LogisticRegression(random_state = 45)
model_4.fit(X_train.values, y_train)

# Calculate model perfomance
predictions = model_4.predict(X_test)
model_4_score = accuracy_score(y_test, predictions)

print('Accuracy score for Logistic regression is', model_4_score)


# In[61]:


class_names = ["non Diabetes","Diabetes"]
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(model_4, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Greens,
                                 normalize=normalize)
    disp.ax_.set_title(title)


# In[62]:


y_pred_train = model_4.predict(X_train)
y_pred_test = model_4.predict(X_test)


# In[63]:


fpr_log,tpr_log,thres_log = roc_curve(y_test, y_pred_test)
log_precision, log_recall, log_thres = precision_recall_curve(y_test, y_pred_test)


fig, ax = plt.subplots(1,2,figsize=(20,6))
ax[0].plot(fpr_log,tpr_log)
ax[0].plot([0, 1], ls="--")
ax[0].plot([0, 0], [1, 0] , c=".7")
ax[0].plot([1, 1] , c=".7")
ax[0].set_ylabel('True Positive Rate')
ax[0].set_xlabel('False Positive Rate')
print(roc_auc_score(y_test, y_pred_test))


ax[1].plot(log_recall,log_precision)
ax[1].plot([0, 1], ls="--")
ax[1].plot([0, 0], [1, 0] , c=".7")
ax[1].plot([1, 1] , c=".7")
ax[1].set_ylabel('Precision')
ax[1].set_xlabel('Recall')
plt.show()


# In[64]:


accuracy_score(y_train, y_pred_train)


# In[65]:


accuracy_score(y_test, y_pred_test)


# In[ ]:





# ## Model 5 - SGD Classifier

# In[66]:


model_5 = SGDClassifier(random_state = 45)
model_5.fit(X_train, y_train)

# Calculate model perfomance
predictions = model_5.predict(X_test)
model_5_score = accuracy_score(y_test, predictions)

print('Accuracy score for sgd is', model_5_score)


# In[67]:


class_names = ["non Diabetes","Diabetes"]
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(model_5, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Greens,
                                 normalize=normalize)
    disp.ax_.set_title(title)


# In[68]:


y_pred_train = model_5.predict(X_train)
y_pred_test = model_5.predict(X_test)


# In[69]:


fpr_log,tpr_log,thres_log = roc_curve(y_test, y_pred_test)
log_precision, log_recall, log_thres = precision_recall_curve(y_test, y_pred_test)


fig, ax = plt.subplots(1,2,figsize=(20,6))
ax[0].plot(fpr_log,tpr_log)
ax[0].plot([0, 1], ls="--")
ax[0].plot([0, 0], [1, 0] , c=".7")
ax[0].plot([1, 1] , c=".7")
ax[0].set_ylabel('True Positive Rate')
ax[0].set_xlabel('False Positive Rate')
print(roc_auc_score(y_test, y_pred_test))


ax[1].plot(log_recall,log_precision)
ax[1].plot([0, 1], ls="--")
ax[1].plot([0, 0], [1, 0] , c=".7")
ax[1].plot([1, 1] , c=".7")
ax[1].set_ylabel('Precision')
ax[1].set_xlabel('Recall')
plt.show()


# In[70]:


accuracy_score(y_train, y_pred_train)


# In[71]:


accuracy_score(y_test, y_pred_test)


# In[ ]:





# ### Results and Comparism

# In[72]:


r1=metrics.recall_score(y_test, model_1.predict(X_test))*100
r2=metrics.recall_score(y_test, model_2.predict(X_test))*100
r3=metrics.recall_score(y_test, model_3.predict(X_test))*100
r4=metrics.recall_score(y_test, model_4.predict(X_test))*100
r5=metrics.recall_score(y_test, model_5.predict(X_test))*100


# In[73]:


p1=metrics.precision_score(y_test, model_1.predict(X_test))*100
p2=metrics.precision_score(y_test, model_2.predict(X_test))*100
p3=metrics.precision_score(y_test, model_3.predict(X_test))*100
p4=metrics.precision_score(y_test, model_4.predict(X_test))*100
p5=metrics.precision_score(y_test, model_5.predict(X_test))*100


# In[74]:


f1=metrics.f1_score(y_test, model_1.predict(X_test))*100
f2=metrics.f1_score(y_test, model_2.predict(X_test))*100
f3=metrics.f1_score(y_test, model_3.predict(X_test))*100
f4=metrics.f1_score(y_test, model_4.predict(X_test))*100
f5=metrics.f1_score(y_test, model_5.predict(X_test))*100


# In[75]:


result=pd.DataFrame({"Model No":[1,2,3,4,5],
                    "Model":["Decision Tree","Decision Tree","K Neighbors","Logistic Regression","SGD"],
                    "Accuracy score":[model_1_score*100,model_2_score*100,model_3_score*100,model_4_score*100,model_5_score*100],
                    "Recall":[r1,r2,r3,r4,r5],"precision":[p1,p2,p3,p4,p5],"F1 Score":[f1,f2,f3,f4,f5]
                    })


# In[76]:


result


# In[ ]:





# In[77]:


plt.figure(figsize=(10, 5)) 
ax = sns.barplot(x='Model', y="Accuracy score", data=result,order=result.sort_values("Accuracy score").Model)
ax.bar_label(ax.containers[0])
ax.set_title('Models Accuracy score')
plt.show()


# In[78]:


plt.figure(figsize=(10, 5)) 
ax = sns.barplot(x='Model', y="Recall", data=result,order=result.sort_values("Recall").Model)
ax.bar_label(ax.containers[0], label_type='edge')
ax.set_title('Models Recall')
plt.show()


# In[79]:


plt.figure(figsize=(10, 5)) 
ax = sns.barplot(x='Model', y="precision", data=result,order=result.sort_values("precision").Model)
ax.bar_label(ax.containers[0], label_type='edge')
ax.set_title('Models precision')
plt.show()


# In[80]:



plt.figure(figsize=(10, 5)) 
ax = sns.barplot(x='Model', y="F1 Score", data=result,order=result.sort_values("F1 Score").Model)
ax.bar_label(ax.containers[0], label_type='edge')
ax.set_title('Models F1 Score')
plt.show()


# ### Saving Model

# In[ ]:


import pickle


# filename = 'LogisticReg.sav'
# pickle.dump(model_4, open(filename, 'wb'))

# ### Conclustion

# In[ ]:




