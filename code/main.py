# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# data road
train_df = pd.read_csv('D:/titanic/input/train.csv')
test_df = pd.read_csv('D:/titanic/input/test.csv')
combine = [train_df, test_df]

# print(train_df.columns.values)

## preview the data
# print(train_df.head())
# print(train_df.tail())

## empty value check
# train_df.info()
# print('_'*40)
# test_df.info()

# print(train_df.describe())

# print(train_df.describe(include=['O'])) #find columns include string

## find columns related survived
# print(train_df[['Pclass', 'Survived']].groupby(['Pclass'],
# as_index=False).mean().sort_values(by='Survived', ascending=False))

# print(train_df[['Sex', 'Survived']].groupby(['Sex'],
# as_index=False).mean().sort_values(by='Survived', ascending=False))

# print(train_df[['SibSp', 'Survived']].groupby(['SibSp'],
# as_index=False).mean().sort_values(by='Survived', ascending=False))

# print(train_df[['Parch', 'Survived']].groupby(['Parch'],
# as_index=False).mean().sort_values(by='Survived', ascending=False))

## Analyze by visualizing data
# Age & survived
# g = sns.FacetGrid(train_df, col='Survived')
# print(g)
# g.map(plt.hist, 'Age', bins=20)
# plt.show()
# Pclass & survived
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
# # grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()
# plt.show()

## Correlating categorical features
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', plaette='deep')
grid.add_legend()
# plt.show()
## Correlating categorical and numerical features
#grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
# plt.show()

## Correcting by dropping features
# print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
# print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

## Creating new feature extracting from existing
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
# print(pd.crosstab(train_df['Title'], train_df['Sex']))

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

title_mapping = {"Mr" : 1, "Miss" : 2, "Mrs" : 3, "Master" : 4, "Rare" : 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0) # fill NaN to zero
# print(train_df.head())
# print(train_df.shape, test_df.shape)

## drop the name and passengerID
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
# print(train_df.shape, test_df.shape)

## Converting a categorical feature
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female' : 1, 'male' : 0}).astype(int)
# print(train_df.head())

## Conpleting a numerical continuous feature
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
# plt.show()

## guess ages to fill the empty space(Sex & Pclass)
guess_ages = np.zeros((2, 3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &
                (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i)
                & (dataset.Pclass == j+1), 'Age'] = guess_ages[i,j]
    dataset['Age'] = dataset['Age'].astype(int)
# print(train_df.head())

train_df['AgeBand'] = pd.cut(train_df['Age'], 5) # divide equal length
# print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
# print(train_df.head())

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
# print(train_df.head())

## Create new feature combining existing features
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'],
#     as_index=False).mean().sort_values(by='Survived', ascending=False))

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'],
#     as_index=False).mean())

## Drop Parch, SibSp, FamilySize
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]
# print(train_df.head())

## new feature by Pclass * Age
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
# print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))

## Completing a categorical feature
freq_port = train_df.Embarked.dropna().mode()[0] #most common value
# print(freq_port)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
# print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False)
# .mean().sort_values(by='Survived', ascending=False))

## Converting categorical feature to numeric
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
# print(train_df.head())

## Quick completing and converting a numeric feature
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
# print(test_df.head())

## Fare feature to ordinal values
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4) # divide equal-size
# print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False)
# .mean().sort_values(by='FareBand', ascending=True))

for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & dataset['Fare'] <= 14.454, 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & dataset['Fare'] <= 31, 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
train_df = train_df.drop(['FareBand'], axis=1)
combind = [train_df, test_df]
# print(train_df.head())
# print(test_df.head())

#### Model, predict and solve
train_data = train_df.drop("Survived", axis=1)
train_label = train_df["Survived"]
test_data = test_df.drop("PassengerId", axis=1).copy()

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()
# print(X_train.shape, Y_train.shape, X_test.shape)

## split train and test dataset
def train_and_test(model, train_data, train_label):
    x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.2, shuffle=True, random_state=5)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    accuracy = round(accuracy_score(y_test, prediction) * 100, 2)
    print("Accuracy : ", accuracy, "%")
    return prediction

## model(mine)
log_pred = train_and_test(LogisticRegression(), train_data, train_label)
svm_pred = train_and_test(SVC(), train_data, train_label)
rf_pred = train_and_test(RandomForestClassifier(n_estimators=100), train_data, train_label)

print(log_pred, svm_pred, rf_pred)

## Logistic Regression
# logreg = LogisticRegression()
# logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(X_test)
# acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# print('acc_log : ', acc_log)

# coeff_df = pd.DataFrame(train_df.columns.delete(0))
# coeff_df.columns = ['Feature']
# print(logreg.coef_)
# coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
# print(coeff_df.sort_values(by='Correlation', ascending=False))

## Support Vector Machines
# svc = SVC()
# svc.fit(X_train, Y_train)
# Y_pred = svc.predict(X_test)
# acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
# print('acc_svc : ', acc_svc)

## KNN
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, Y_train)
# Y_pred = knn.predict(X_test)
# acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
# print('acc_knn : ', acc_knn)

# Gaussian Naive Bayes
# gaussian = GaussianNB()
# gaussian.fit(X_train, Y_train)
# Y_pred = gaussian.predict(X_test)
# acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
# print('acc_gaussian : ', acc_gaussian)

# Perceptron
# perceptron = Perceptron()
# perceptron.fit(X_train, Y_train)
# Y_pred = perceptron.predict(X_test)
# acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
# print('acc_perceptron : ', acc_perceptron)

# Linear SVC
# linear_svc = LinearSVC()
# linear_svc.fit(X_train, Y_train)
# Y_pred = linear_svc.predict(X_test)
# acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
# print('acc_linear_svc : ', acc_linear_svc)

# Stochastic Gradient Descent
# sgd = SGDClassifier()
# sgd.fit(X_train, Y_train)
# Y_pred = sgd.predict(X_test)
# acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
# print('acc_sgd : ', acc_sgd)

# Decision Tree
# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(X_train, Y_train)
# Y_pred = decision_tree.predict(X_test)
# acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
# print('acc_decision_tree : ', acc_decision_tree)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print('acc_random_forest : ', acc_random_forest)

## Model evaluation
#decision_tree or random forest
# models = pd.DataFrame({
#     'Model' : ['Support Vector Machines', 'KNN', 'Logistic Regression',
#                 'Random Forest', 'Naive Bayes', 'Perceptron',
#                 'Stochastic Gradient Decent', 'Linear SVC', 'Decision Tree'],
#     'Score' : [acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian,
#                 acc_perceptron, acc_sgd, acc_linear_svc, acc_decision_tree]
# })
# print(models.sort_values(by='Score', ascending=False))

submission = pd.DataFrame({
    "PassengerId" : test_df["PassengerId"],
    "Survived" : Y_pred
})
submission.to_csv('D:/titanic/output/submission.csv', index=False)
