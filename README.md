# 𝐌𝐈𝐃𝐓𝐄𝐑𝐌: 𝐏𝐚𝐢𝐫-𝐁𝐚𝐬𝐞𝐝 𝐏𝐫𝐨𝐣𝐞𝐜𝐭 𝐛𝐲 DE CASTRO & OXILLO

<p align="center">
  <img src="https://github.com/kenziruss/Oxillo_De-Castro_Adidas-US-Sale-Datasets/blob/main/Blue%20Pink%20Pixelation%20Coming%20Soon%20Video.gif">
</p>


## Group Information
```
Group Name Members: De Castro, Rod Andrei & Oxillo, Ivan Russ
Assigned Theme:
Topic: Adidad US Sale Datasets
```
## Project Overview
Dataset Title: Adidad US Sale Datasets
```
```
## Source 
 ```
The Adidas Sales Dataset was published on Kaggle by Heemali Chaudhari. It contains sales transaction records for Adidas products, and is available for download on the Kaggle platform (Chaudhari, n.d.). https://www.kaggle.com/datasets/heemalichaudhari/adidas-sales-dataset
   ```
```
 ```
## License/Attribution:
```
```
## Objective:
 ```
To analyze the dataset through a thorough exploration of its structure, including the identification of missing values, outliers, and inconsistencies. This process also involves determining the necessary preprocessing steps such as data cleaning, transformation, and preparation for further analysis.
```
 ```
```
## Implementation Guide (Step-by-Step Process)
### Data Preprocessing
 ```
### Step 1. Data Processing - Importing Libraries and Loading Dataset
* Import essential libraries such as **pandas** for data handling and **scikit-learn** modules (`StandardScaler`, `OneHotEncoder`, `SimpleImputer`, `ColumnTransformer`, `Pipeline`) for preprocessing.  
* Load the dataset **(Adidas US Sales Datasets.xlsx)** into a pandas DataFrame for further analysis.
 ```
### 1. Data Processing
```
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

```
### Load the dataset
```
df = pd.read_excel('Adidas US Sales Datasets.xlsx')

# Display dataset overview
display(df.head())
display(df.describe())
df.info()

```
### 2. Defining Preprocessing Steps
```
# Identify numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Create a pipeline for numerical features
# Impute missing values with the median and then apply StandardScaler
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

```

```
# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

# Create a pipeline for categorical features
# Impute missing values with a constant 'missing' and then apply OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
```

### 3: Combining Preprocessing Steps
```
# Create a ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
```
 ### 4: Applying the Preprocessing Pipeline
 ```
# Apply the preprocessing to the data
df_preprocessed = preprocessor.fit_transform(df)

# Display the shape of the preprocessed data
display(df_preprocessed.shape)
 ```
 ```
print(df_preprocessed)
 ```
 ```
# Discretize 'Total Sales' into 5 bins
df['Total Sales_Bin'] = pd.cut(df['Total Sales'], bins=5)

# Display the first few rows with the new discretized column
display(df[['Total Sales', 'Total Sales_Bin']].head())

# Display the value counts for the new discretized column
display(df['Total Sales_Bin'].value_counts())
 ```

## Step 2: Analyzing the Preprocessed Data
### 1. Data Quality Report
 ```
print("\nMissing values after preprocessing:")
display(df_preprocessed_dense.isnull().sum())
 ```
 ### 2. Visualization
  ```
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the histogram of the original 'Total Sales'
plt.figure(figsize=(8, 5))
sns.histplot(df['Total Sales'], kde=True)
plt.title('Distribution of Total Sales')
plt.xlabel('Total Sales')
plt.ylabel('Frequency')
plt.show()
```
```
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the KDE of 'Operating Profit'
plt.figure(figsize=(8, 5))
sns.histplot(df['Operating Profit'], kde=True)
plt.title('KDE Plot – Profit Density')
plt.xlabel('Operating Profit')
plt.ylabel('Density')
plt.show()
```
```
import matplotlib.pyplot as plt
import seaborn as sns

# Create a scatter plot of 'Price per Unit' vs 'Units Sold'
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Price per Unit', y='Units Sold')
plt.title('Scatter Plot – Price vs Units Sold')
plt.xlabel('Price per Unit')
plt.ylabel('Units Sold')
plt.show()
```
```
import matplotlib.pyplot as plt
import seaborn as sns

# Create a pairplot for the specified numerical columns
numerical_vars = ['Total Sales', 'Operating Profit', 'Price per Unit', 'Units Sold']
sns.pairplot(df[numerical_vars])
plt.suptitle('Pairplot – Sales, Profit, Price, Units', y=1.02) # Add a title to the pairplot
plt.show()
```
```
import matplotlib.pyplot as plt
import seaborn as sns

# Select the numerical columns for the heatmap
business_metrics = ['Total Sales', 'Operating Profit', 'Price per Unit', 'Units Sold']
correlation_matrix = df[business_metrics].corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap – Business Metrics')
plt.show()
```
```
!pip install missingno
```
```
import missingno as msno

# Create a missingness matrix plot for the original data DataFrame
msno.matrix(data)
plt.title('Missingness Matrix of Original Data')
plt.show()
```
```
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the average sales by region
average_sales_by_region = df.groupby('Region')['Total Sales'].mean().reset_index()

# Create a bar plot of average sales by region
plt.figure(figsize=(10, 6))
sns.barplot(data=average_sales_by_region, x='Region', y='Total Sales')
plt.title('Average Sales by Region')
plt.xlabel('Region')
plt.ylabel('Average Total Sales')
plt.xticks(rotation=45, ha='right') # Rotate labels for readability
plt.tight_layout()
plt.show()

```
```
import matplotlib.pyplot as plt
import seaborn as sns

# Create a box plot of 'Operating Profit' by 'Sales Method'
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Sales Method', y='Operating Profit')
plt.title('Box Plot – Profit by Sales Method')
plt.xlabel('Sales Method')
plt.ylabel('Operating Profit')
plt.show()
```
```
import matplotlib.pyplot as plt
import seaborn as sns

# Resample data by week and sum 'Total Sales'
weekly_sales = df.set_index('Invoice Date')['Total Sales'].resample('W').sum().reset_index()

# Create a line plot of weekly sales trend
plt.figure(figsize=(12, 6))
sns.lineplot(data=weekly_sales, x='Invoice Date', y='Total Sales')
plt.title('Weekly Sales Trend')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()
```
```
# Identify the names of the columns in the df DataFrame that contain numerical data.
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Display the list of numerical features
print("Numerical features selected for PCA:")
print(numerical_features)
```
```
from sklearn.decomposition import PCA

# Instantiate PCA with 2 components
pca = PCA(n_components=2)

# Fit PCA to the numerical features and transform the data
principal_components = pca.fit_transform(df[numerical_features])

# Display the shape of the resulting principal components
display(principal_components.shape)
```
```
# Create a DataFrame from the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Add the 'Region' column from the original DataFrame
pca_df['Region'] = df['Region']

# Display the head of the new DataFrame
display(pca_df.head())
```
```
# Create a scatter plot of PC1 vs PC2, colored by Region
plt.figure(figsize=(10, 7))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Region')
plt.title('PCA of Numerical Features by Region')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
```
```

