#!/usr/bin/env python
# coding: utf-8

# # 1: Exploratory data analysis
# 
# We start by looking at how to explore our data. We will cover
# 
# * Categorical variables
# * Numeric variables
# * Looking for outliers
# * Exploring correlations in the data
# 
# Some extrnal resources that you might find useful:
# 
# * https://towardsdatascience.com/exploratory-data-analysis-in-python-a-step-by-step-process-d0dfa6bf94ee

# In[1]:


# data manipulation
import pandas as pd
import numpy as np

# data viz
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# apply some cool styling
plt.style.use("ggplot")
rcParams['figure.figsize'] = (12,  6)

# use sklearn to import a dataset
from sklearn.datasets import load_wine


# ## Import the data
# Importing a dataset is simple with Pandas through functions dedicated to reading the data. If our dataset is a .csv file, we can just use
# 
# `df = pd.read_csv("path/to/my/file.csv")`
# 
# df stands for dataframe, which is Pandas’s object similar to an Excel sheet. This nomenclature is often used in the field. The read_csv function takes as input the path of the file we want to read. There are many other arguments that we can specify.
# 
# The .csv format is not the only one we can import — there are in fact many others such as Excel, Parquet and Feather.
# 
# For ease, in this example we will use Sklearn to import the wine dataset.
# 
# We eill set a new column called `target` which is a class for the wines.

# In[3]:


# carichiamo il dataset
wine = load_wine()

# convertiamo il dataset in un dataframe Pandas
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
# creiamo la colonna per il target
df["target"] = wine.target


# ## Take a quick look using Pandas
# 
# Two of the most commonly used functions in Pandas are `.head()` and `.tail()`. These two allow us to view an arbitrary number of rows (by default 5) from the beginning or end of the dataset. Very useful for accessing a small part of the dataframe quickly.
# 
# We can also use:
# * `.shape`
# * `.describe()`
# * `.info()`
# 
# **Note** the `.shape` call is not followed by parentheses `()`; this is beacause `shape` is an attribute of the dataset, whereas `describe` for example is a function that acts on the dataset.

# In[4]:


df.tail(3)


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.info()


# Notice that info gives quite different results to describe. Info tells us about the data types - describe gives us some summary statistics.

# ## Make names more sensible
# 
# The name `od280/od315_of_diluted_wines` refers to a test for protein content. But it is not very descriptive to us, let's chance it to make life easier for ourselves later.

# In[8]:


df.rename(columns={"od280/od315_of_diluted_wines": "protein_concentration"}, inplace=True)


# ## Undestanding the variables
# 
# We will look at two main types of variable discussed in the lecture: categorical and numeric.
# 
# ### Categorical variables
# 
# Categorical variables are those where the data are labelled by class, for example it could be data on something like postcode. If we had labelled houses by postcode, then this is said to be categorical data.
# 
# Let's look at the distribution of the types for the wines, the `target` column
# 

# In[9]:


df.target.value_counts()


# In[10]:


df.target.value_counts(normalize=True)


# In[16]:


df.target.value_counts().plot(kind="bar")
plt.title("Value counts of the target variable")
plt.xlabel("Wine type")
plt.xticks(rotation=0)
plt.ylabel("Count")
plt.show()


# ### Numeric values
# 
# Numeric data is when we assign a numerical value as the label of an instance. To take the houses example again, we might label the houses by the distance to the nearest bus stop, this would then be a numeric data set.
# 
# We can perform exploratory analysis of the numeric values using the `.describe()` function.

# In[17]:


df.magnesium.describe()


# In[18]:


df.magnesium.hist()


# Question do you think this has high/low skew - or high/low kurotis?

# In[11]:


print(f"Skewness: {df['magnesium'].skew()}")
print(f"Kurtosis: {df['magnesium'].kurt()}")


# In[12]:


sns.catplot(x="target", y="proline", data=df, kind="box", aspect=1.5)
plt.title("Boxplot for target vs proline")
plt.show()


# ## Question which of the datasets might have an outlier

# In[27]:


sns.catplot(x="target", y="flavanoids", data=df, kind="box", aspect=1.5)
plt.title("Boxplot for target vs proline")
plt.show()


# ## Take a closer look at the outlier

# In[25]:


df[df['flavanoids'] > 5]


# We can then remove that data from the data set, if we choose to.

# In[26]:


df = df.drop(121)


# ## Explore correlations in the data
# 
# As we discussed in the lecture we might want to remove features in the data that are very closely related to one another. Imagine a data set that had one feature with temperature in F and one with temperature in C. Now these two features tell us exactly the same thing, but on a different scale. Later we will see that adding both to a model would be of no benifit, as they do not carry any extra information. To look for redundant information between features ($x$ and $y$) we can use the correlation.
# $$
# r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2(y_i - \bar{y})^2}}
# $$
# 

# In[23]:


corrmat = df.corr()
corrmat


# In[24]:


hm = sns.heatmap(corrmat, 
                 cbar=True, 
                 annot=True, 
                 square=True, 
                 fmt='.2f', 
                 annot_kws={'size': 10}, 
                 yticklabels=df.columns, 
                 xticklabels=df.columns, 
                 cmap="Spectral_r")
plt.show()


# Question - which data are most correlated. If you had to choose to drop one column based on the relation to the target, which would it be? If you had to drop one of the strongly correlated columns, which would it be? 

# In[ ]:




