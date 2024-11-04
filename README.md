# User Behavior Analysis Project

## Introduction
This project aims to analyze user behavior data to uncover insights related to screen time, app usage, and battery drain. The analysis focuses on various demographics and operating systems to provide a comprehensive overview of user engagement.

## Table of Contents
1. [Import Libraries and Load the Dataset](#import-libraries-and-load-the-dataset)
2. [Data Cleaning and Exploration](#data-cleaning-and-exploration)
3. [Answering Key Questions with Visualizations](#answering-key-questions-with-visualizations)
   - [3.1 What Age and Gender Use the Most Screen Time?](#31-what-age-and-gender-use-the-most-screen-time)
   - [3.2 Which Operating System Has the Most App Usage Time and Screen On Time?](#32-which-operating-system-has-the-most-app-usage-time-and-screen-on-time)
   - [3.3 Find the Correlation Between Screen On Time and Battery Drain](#33-find-the-correlation-between-screen-on-time-and-battery-drain)
4. [Save Cleaned Data for Tableau Visualization](#save-cleaned-data-for-tableau-visualization)

## Step 1: Import Libraries and Load the Dataset
In this step, we import the necessary libraries and load the user behavior dataset.



```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load the dataset
data = pd.read_csv('user_behavior_dataset.csv')

# Display the first few rows
data.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User ID</th>
      <th>Device Model</th>
      <th>Operating System</th>
      <th>App Usage Time (min/day)</th>
      <th>Screen On Time (hours/day)</th>
      <th>Battery Drain (mAh/day)</th>
      <th>Number of Apps Installed</th>
      <th>Data Usage (MB/day)</th>
      <th>Age</th>
      <th>Gender</th>
      <th>User Behavior Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Google Pixel 5</td>
      <td>Android</td>
      <td>393</td>
      <td>6.4</td>
      <td>1872</td>
      <td>67</td>
      <td>1122</td>
      <td>40</td>
      <td>Male</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>OnePlus 9</td>
      <td>Android</td>
      <td>268</td>
      <td>4.7</td>
      <td>1331</td>
      <td>42</td>
      <td>944</td>
      <td>47</td>
      <td>Female</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Xiaomi Mi 11</td>
      <td>Android</td>
      <td>154</td>
      <td>4.0</td>
      <td>761</td>
      <td>32</td>
      <td>322</td>
      <td>42</td>
      <td>Male</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Google Pixel 5</td>
      <td>Android</td>
      <td>239</td>
      <td>4.8</td>
      <td>1676</td>
      <td>56</td>
      <td>871</td>
      <td>20</td>
      <td>Male</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>iPhone 12</td>
      <td>iOS</td>
      <td>187</td>
      <td>4.3</td>
      <td>1367</td>
      <td>58</td>
      <td>988</td>
      <td>31</td>
      <td>Female</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Checking the number of rows and columns in the dataset
data.shape
```




    (700, 11)




```python
# Check the columns names
data.columns.tolist()
```




    ['User ID',
     'Device Model',
     'Operating System',
     'App Usage Time (min/day)',
     'Screen On Time (hours/day)',
     'Battery Drain (mAh/day)',
     'Number of Apps Installed',
     'Data Usage (MB/day)',
     'Age',
     'Gender',
     'User Behavior Class']



## Step 2: Data Cleaning and Exploration

In this step we check for missing values, remove any uncessesary columns, and check data types (convert if necessary). Then we get a quick statisicaly summary. 


```python
# Check the data types to make sure they are correct
data.dtypes
```




    User ID                         int64
    Device Model                   object
    Operating System               object
    App Usage Time (min/day)        int64
    Screen On Time (hours/day)    float64
    Battery Drain (mAh/day)         int64
    Number of Apps Installed        int64
    Data Usage (MB/day)             int64
    Age                             int64
    Gender                         object
    User Behavior Class             int64
    dtype: object




```python
# Check for missing values
data.isnull().sum()
```




    User ID                       0
    Device Model                  0
    Operating System              0
    App Usage Time (min/day)      0
    Screen On Time (hours/day)    0
    Battery Drain (mAh/day)       0
    Number of Apps Installed      0
    Data Usage (MB/day)           0
    Age                           0
    Gender                        0
    User Behavior Class           0
    dtype: int64




```python
# Make sure there are no duplicates
data.duplicated().sum()
```




    0




```python
# Drop unneeded columns
data.drop(['User ID','User Behavior Class'], axis=1, inplace=True)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Device Model</th>
      <th>Operating System</th>
      <th>App Usage Time (min/day)</th>
      <th>Screen On Time (hours/day)</th>
      <th>Battery Drain (mAh/day)</th>
      <th>Number of Apps Installed</th>
      <th>Data Usage (MB/day)</th>
      <th>Age</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Google Pixel 5</td>
      <td>Android</td>
      <td>393</td>
      <td>6.4</td>
      <td>1872</td>
      <td>67</td>
      <td>1122</td>
      <td>40</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OnePlus 9</td>
      <td>Android</td>
      <td>268</td>
      <td>4.7</td>
      <td>1331</td>
      <td>42</td>
      <td>944</td>
      <td>47</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Xiaomi Mi 11</td>
      <td>Android</td>
      <td>154</td>
      <td>4.0</td>
      <td>761</td>
      <td>32</td>
      <td>322</td>
      <td>42</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Google Pixel 5</td>
      <td>Android</td>
      <td>239</td>
      <td>4.8</td>
      <td>1676</td>
      <td>56</td>
      <td>871</td>
      <td>20</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>4</th>
      <td>iPhone 12</td>
      <td>iOS</td>
      <td>187</td>
      <td>4.3</td>
      <td>1367</td>
      <td>58</td>
      <td>988</td>
      <td>31</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Device Model</th>
      <th>Operating System</th>
      <th>App Usage Time (min/day)</th>
      <th>Screen On Time (hours/day)</th>
      <th>Battery Drain (mAh/day)</th>
      <th>Number of Apps Installed</th>
      <th>Data Usage (MB/day)</th>
      <th>Age</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>700</td>
      <td>700</td>
      <td>700.000000</td>
      <td>700.000000</td>
      <td>700.000000</td>
      <td>700.000000</td>
      <td>700.000000</td>
      <td>700.000000</td>
      <td>700</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>5</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Xiaomi Mi 11</td>
      <td>Android</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>146</td>
      <td>554</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>364</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>271.128571</td>
      <td>5.272714</td>
      <td>1525.158571</td>
      <td>50.681429</td>
      <td>929.742857</td>
      <td>38.482857</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>177.199484</td>
      <td>3.068584</td>
      <td>819.136414</td>
      <td>26.943324</td>
      <td>640.451729</td>
      <td>12.012916</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>302.000000</td>
      <td>10.000000</td>
      <td>102.000000</td>
      <td>18.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>113.250000</td>
      <td>2.500000</td>
      <td>722.250000</td>
      <td>26.000000</td>
      <td>373.000000</td>
      <td>28.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>227.500000</td>
      <td>4.900000</td>
      <td>1502.500000</td>
      <td>49.000000</td>
      <td>823.500000</td>
      <td>38.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>434.250000</td>
      <td>7.400000</td>
      <td>2229.500000</td>
      <td>74.000000</td>
      <td>1341.000000</td>
      <td>49.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>598.000000</td>
      <td>12.000000</td>
      <td>2993.000000</td>
      <td>99.000000</td>
      <td>2497.000000</td>
      <td>59.000000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Step 3: Answering Key Questions with Visualizations

### 3.1 What Age and Gender Use the Most Screen Time?

We wanted to check the screen time by age to see if anything stood out.


```python
# Average Screen On Time by Age
screen_time_by_age = data.groupby('Age')['Screen On Time (hours/day)'].mean()

# Plotting
plt.figure(figsize=(10, 6), facecolor='lightgrey')
plt.gca().set_facecolor('grey')
screen_time_by_age.plot(kind='bar', color='orange')
plt.title('Average Screen On Time by Age')
plt.xlabel('Age')
plt.ylabel('Screen On Time (hours/day)')
plt.show()
```


    
![png](user_behavior_files/user_behavior_11_0.png)
    


Next we check screen time by Gender


```python
# Average Screen On Time by Gender
screen_time_by_gender = data.groupby('Gender')['Screen On Time (hours/day)'].mean()

# Plotting
plt.figure(figsize=(8, 5), facecolor='lightgrey')
plt.gca().set_facecolor('grey')
screen_time_by_gender.plot(kind='bar', color=['#ff9999','#66b3ff'])
plt.yticks(np.arange(0, 6, step=0.5))
plt.title('Average Screen On Time by Gender')
plt.xlabel('Gender')
plt.ylabel('Screen On Time (hours/day)')

plt.xticks(rotation=0)
plt.show()
```


    
![png](user_behavior_files/user_behavior_13_0.png)
    



```python
# Comfirming that both Genders are roughly the same 
data[data['Gender'] == 'Male']['Screen On Time (hours/day)'].mean()
```




    5.283241758241759




```python
data[data['Gender'] == 'Female']['Screen On Time (hours/day)'].mean()
```




    5.2613095238095235



### 3.2 Which Operating System Has the Most App Usage Time and Screen On Time?

Next we want to check App Usage time and Screen On Time for each Operating System.


```python
# Average App Usage Time by OS
app_usage_by_os = data.groupby('Operating System')['App Usage Time (min/day)'].mean()

# Plotting
plt.figure(figsize=(10, 6), facecolor='lightgrey')
plt.gca().set_facecolor('grey')
app_usage_by_os.plot(kind='bar', color='salmon')
plt.yticks(np.arange(0, 300, step=10))
plt.title('Average App Usage Time by Operating System')
plt.xlabel('Operating System')
plt.ylabel('App Usage Time (min/day)')
plt.show()

```


    
![png](user_behavior_files/user_behavior_17_0.png)
    



```python
# Seeing specifically how close the two operating systems are from the previous chart 
data[data['Operating System'] == 'Android']['App Usage Time (min/day)'].mean()
```




    268.2581227436823




```python
data[data['Operating System'] == 'iOS']['App Usage Time (min/day)'].mean()
```




    282.02054794520546




```python
# Average Screen On Time by OS
screen_time_by_os = data.groupby('Operating System')['Screen On Time (hours/day)'].mean()

# Plotting
plt.figure(figsize=(10, 6), facecolor='lightgrey')
plt.gca().set_facecolor('grey')
plt.yticks(np.arange(0, 6, step=.25))
screen_time_by_os.plot(kind='bar', color='lightgreen')
plt.title('Average Screen On Time by Operating System')
plt.xlabel('Operating System')
plt.ylabel('Screen On Time (hours/day)')
plt.show()

```


    
![png](user_behavior_files/user_behavior_20_0.png)
    


### 3.3 Find the Correlation Between Screen On Time and Battery Drain

Next we use Pearson correlation to measure the strength and direction of the relationship. Then we graph a scatter plot with a regression line to visualize the correlation.


```python
# Calculate correlation
correlation, p_value = pearsonr(data['Screen On Time (hours/day)'], data['Battery Drain (mAh/day)'])
print(f'Correlation between Screen On Time and Battery Drain: {correlation:.2f}')

```

    Correlation between Screen On Time and Battery Drain: 0.95
    


```python
# Scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x='Screen On Time (hours/day)', y='Battery Drain (mAh/day)', data=data, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Correlation between Screen On Time and Battery Drain')
plt.xlabel('Screen On Time (hours/day)')
plt.ylabel('Battery Drain (mAh/day)')
plt.show()

```


    
![png](user_behavior_files/user_behavior_23_0.png)
    


## Step 4: Save Cleaned Data for Tableau Visualization

Finally, we save the cleaned dataset and save it for futher Visualization in Tableau.


```python
data.to_csv('cleaned_user_behaviour_data.csv', index=False)
```
