import pandas as pd, numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import neighbors
import statistics
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data11.csv', encoding='ISO-8859-1')
print("The data dimentions is: ", data.shape)
# Data exploration
data_des = data.describe()
countNull = data.isnull().sum()

# If there are more than third empty values, cut the column
k = 0
while k < data.shape[1]:
    if countNull.iloc[k] > data.shape[0] / 3:
        title = countNull.index[k]
        data.drop([title], axis=1, inplace=True)
    k += 1

print("The data dimentions after removing NA's is : ", data.shape)

# If there are only one value at the column drop it
dataUnique = data.nunique()
useless_column = dataUnique[dataUnique == 1].index
data = data.drop(useless_column, axis=1)

# We added a BMI column which is basically a common calculator built on a 
# person’s weight and height

data['BMI']= data['wt'] / (pow((data['ht']/100),2))
# Create a range column of BMI
# Underweight  
data['BMI_levels']= np.where(data['BMI'] < 18.5, 0,data['BMI'])
# A proper weight
data['BMI_levels']= np.where(data['BMI'].between(18.5,25) , 1, data['BMI_levels'])
# Over-weight
data['BMI_levels']= np.where(data['BMI'].between(25,30) , 2, data['BMI_levels'])
# Obesity
data['BMI_levels']= np.where(data['BMI'] > 30, 3, data['BMI_levels'])
data['BMI_levels'] = data['BMI_levels'].astype(int)

print('*'*35)
print(data.isnull().sum())
print('*' * 35)
# Features that removed
delete_features = ['exercise','alcoholfreq','age','alcoholhowmuch','death','hbp','diabetes','pica','smokeyrs', 'BMI' ,'hbpmed','ht','wt','school']
for i in data.columns:
    if i in delete_features:
        data.drop(i, axis=1, inplace=True)

# find the most frequent value
for column in data.columns:
    if column == 'birthplace':
        freq = data[column].value_counts().idxmax()
        data[column] = data[column].fillna(freq)
    else:
        mean_value = data[column].mean()
        data[column] = data[column].fillna(mean_value)

# A unified feature for rare diseases, if the patient has at least one of these diseases- 1
At_least_one_disease=[0] * 1629
Disease_lst=['hayfever','bronch','hf','colitis','pepticulcer','polio','tumor','lackpep','weakheart','nervousbreak','hepatitis','tb','asthma', 'chroniccough','nerves']
for i in Disease_lst:
    At_least_one_disease= At_least_one_disease+data[i]

data['At_least_one_disease']  =At_least_one_disease  
counter=0
for i in  data['At_least_one_disease']:
    data['At_least_one_disease']= np.where(data['At_least_one_disease']>1, 1, data['At_least_one_disease'])
    if i==1:
        counter=counter +1
# Removal of the diseases which we have united
for i in data.columns:
    if i in Disease_lst:
        data.drop(i, axis=1, inplace=True)


# The "place of birth" column is numeric and we do not want these numbers to be meaningful,
# so we will convert data in this feature into categories.
data["birthplace"] = data.birthplace.astype('category')

# Separate the data into 2 types of variables - categorical and numerical.
# While calculting the corrlations we will use the coressponding corrleation method:
k=0
numerical_columns = list()
categorical_columns = list()

for i in data.dtypes:
    if i == float or statistics.mean(data.iloc[:,k])>10 and data.columns[k] != 'birthplace':
        numerical_columns.append(data.columns[k])
        k+=1  
    else:
        categorical_columns.append(data.columns[k])
        k+=1

data_numerical = pd.DataFrame(data[numerical_columns])
data_cetegorical = pd.DataFrame(data[categorical_columns])

from scipy import stats
# In order to remove outliers and normalize the data in the best way,
# we will check if the data is distributed normally
# Kolmogorov-Smirnov test:
# H0 = The sample comes from a normal distribution.
# H1 =The sample is not coming from normal distribution
alpha=0.05
for i in data_numerical.columns:
    stats.kstest(data_numerical[i], 'norm', alternative='less')
    stats.kstest(stats.norm.rvs(size=100), stats.norm.cdf)
    statistic,pvalue = stats.kstest(data_numerical[i], 'norm', alternative='greater')
    if pvalue <= alpha:
        print("Rejecting of H0 hypothesis. " ,i)
    else:
        print("Acceptance of H0 hypothesis. " ,i)

# After testing we saw that all the numerical data is normally distributed
# Normalize data , standardization :
scaler = StandardScaler()
scaler.fit(data_numerical)
data_numerical = scaler.transform(data_numerical)
data_numerical =pd.DataFrame(data = data_numerical, columns = numerical_columns)

# Remove outliers:
# We chose to remove outliers in numerical data only since we cannot know
# by our specific data, whether there are data interruptions or not.
import warnings
from scipy import stats
for Feature in data_numerical.columns:
    upper_limit = data_numerical[Feature].mean() + 3*data_numerical[Feature].std()
    lower_limit = data_numerical[Feature].mean() - 3*data_numerical[Feature].std()
    data_numerical[(data_numerical[Feature] > upper_limit) | (data_numerical[Feature] < lower_limit)]
    new_df = data_numerical[(data_numerical[Feature] < upper_limit) & (data_numerical[Feature] > lower_limit)]
    new_df
    data_numerical[Feature] = np.where(data_numerical[Feature]>upper_limit, upper_limit, np.where(
            data_numerical[Feature]<lower_limit, lower_limit, data_numerical[Feature]
        )
    )
    warnings.filterwarnings('ignore')
    plt.figure(figsize=(16,5))
    plt.subplot(1,2,1)
    sns.distplot(new_df[Feature])
    plt.show()
    
# We have presented plots of the categorical data in order
# to better understand them
for i in data_cetegorical.columns:
    sns.catplot(x=i, kind="count", palette="ch:.25", data=data)

# Pearson corelation
corr_pearson = data_numerical.corr()
sns.heatmap(corr_pearson)

# Spearman corelation
corr_spearman = data_cetegorical.corr(method="spearman")
sns.heatmap(corr_spearman)

data1 = pd.concat([data_numerical, data_cetegorical], axis=1, join='inner')

features_=data1.iloc[:,data1.columns != 'At_least_one_disease']
target = data_cetegorical['At_least_one_disease']
X_train, X_test, y_train, y_test = train_test_split(features_, target, test_size = 0.1, random_state = 0)

def Logistic_Regreesion(features_ , target):  
    
    ########################## Logistic Regreesion ##########################
    print("Logistic Regreesion Results : ")
    reg = LogisticRegression()
    model1 = reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    # Calculating the confidence level of the prediction
    pred_proba = reg.predict_proba(X_test)
    # Tests if the training is good
    y_pred_tr = reg.predict(X_train)
    error = (y_pred-y_test)
    
    # Check that the model handles the data correctly and that there is a 
    # distribution to the level of security
    plt.figure(figsize=(16,5))
    plt.subplot(1,2,1)
    sns.distplot(pred_proba)
    plt.show()
    # print confusion matrix
    print(metrics.confusion_matrix(y_train, y_pred_tr))
    print(metrics.confusion_matrix(y_test, y_pred))
    # accuracy , recall, precision
    print(classification_report(y_train,y_pred_tr))
    print(classification_report(y_test,y_pred))
    
    intercept_logit =  model1.intercept_[0]
    classes_logit = model1.classes_
    coeff_logit = pd.DataFrame({'coeff': model1.coef_[0]}, index=X_train.columns)
    
from sklearn.inspection import permutation_importance   
from sklearn.neighbors import KNeighborsClassifier 

def KNeighbors_Regression(features_ , target):     
    
    ########################## KNeighbors Regression ##########################
    # Error rate for different k values
    rmse_val = [] 
    for K in range(20):
        K = K+1
        model = neighbors.KNeighborsRegressor(n_neighbors = K)
        model.fit(X_train, y_train)  #fit the model
        pred=model.predict(X_test) #make prediction on test set
        error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
        rmse_val.append(error) #store rmse values
        print('RMSE value for k= ' , K , 'is:', error)
    curve = pd.DataFrame(rmse_val) #elbow curve 
    curve.plot()
    
    # It can be seen in the graph that a change occurs when K=6 , therefore we choose k=6
    # even though RMSE value is not the smallest, the difference in errors is not
    # too big and the result will be better
    
    Classifier = KNeighborsClassifier(n_neighbors=6)
    model2 = Classifier.fit(X_train, y_train)
    y_pred_knn = Classifier.predict(X_test)
    # Calculating the confidence level of the prediction
    pred_proba_knn = Classifier.predict_proba(X_test)
    # Tests if the training is good
    y_pred_tr_knn = Classifier.predict(X_train)
    
    # Check that the model handles the data correctly and that there is a 
    # distribution to the level of security
    plt.figure(figsize=(16,5))
    plt.subplot(1,2,1)
    sns.distplot(pred_proba_knn)
    plt.show()
    print("KNeighbors Regression Results : ")
    #print confusion matrix
    print(metrics.confusion_matrix(y_train, y_pred_tr_knn))
    print(metrics.confusion_matrix(y_test, y_pred_knn))
    #accuracy , recall, precision
    print(classification_report(y_train,y_pred_tr_knn))
    print(classification_report(y_test,y_pred_knn))
 
    # perform permutation importance
    results = permutation_importance(model2, features_, target, scoring='neg_mean_squared_error')
    # get importance
    importance = results.importances_mean
    # summarize feature importance
    for i,v in enumerate(importance):
    	print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()

from sklearn import tree
def Tree_Classifier(features_ , target): 
    
    ##########################Decision Tree Classifier##########################
    print("Tree Classifier Results : ")     
    clf = tree.DecisionTreeClassifier(min_samples_split=30,min_samples_leaf=30,max_features="log2",max_leaf_nodes=20)
    model4 = clf.fit(X_train, y_train)
    y_pred_tree = clf.predict(X_test)
    # Calculating the confidence level of the prediction
    pred_proba_tree = clf.predict_proba(X_test)
    # Tests if the training is good
    y_pred_tr_tree = clf.predict(X_train)
    
    plt.figure(figsize=(16,5))
    plt.subplot(1,2,1)
    sns.distplot(pred_proba_tree)
    plt.show()
    #print confusion matrix
    print(metrics.confusion_matrix(y_train, y_pred_tr_tree))
    print(metrics.confusion_matrix(y_test, y_pred_tree))
    #accuracy , recall, precision
    print(classification_report(y_train,y_pred_tr_tree))
    print(classification_report(y_test,y_pred_tree))
    
    #Print the coefficients for Tree Classifier
    importance = model4.feature_importances_
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()
    

Logistic_Regreesion(features_ , target)
KNeighbors_Regression(features_ , target)
Tree_Classifier(features_ , target)


