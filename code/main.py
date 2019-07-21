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

train_df = pd.read_csv('D:/titanic/input/train.csv')
test_df = pd.read_csv('D:/titanic/input/test.csv')
combine = [train_df, test_df]

# print(train_df.columns.values)

# preview the data
# print(train_df.head())
# print(train_df.tail())

# empty value check
# train_df.info()
# print('_'*40)
# test_df.info()

# print(train_df.describe())

# print(train_df.describe(include=['O'])) #find columns include string

# 생존과 연관성있는 column 찾기
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'],
as_index=False).mean().sort_values(by='Survived', ascending=False))

print(train_df[['Sex', 'Survived']].groupby(['Sex'],
as_index=False).mean().sort_values(by='Survived', ascending=False))

print(train_df[['SibSp', 'Survived']].groupby(['SibSp'],
as_index=False).mean().sort_values(by='Survived', ascending=False))

print(train_df[['Parch', 'Survived']].groupby(['Parch'],
as_index=False).mean().sort_values(by='Survived', ascending=False))

# Analyze by visualizing data
# Age & survived
# g = sns.FacetGrid(train_df, col='Survived')
# print(g)
# g.map(plt.hist, 'Age', bins=20)
# plt.show()
# Pclass & survived
grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
# grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()

# Correlating categorical features
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', plaette='deep')
grid.add_legend()
plt.show()
# Correlating categorical and numerical features
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
plt.show()