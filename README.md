# 𝐌𝐈𝐃𝐓𝐄𝐑𝐌: 𝐏𝐚𝐢𝐫-𝐁𝐚𝐬𝐞𝐝 𝐏𝐫𝐨𝐣𝐞𝐜𝐭 𝐛𝐲 DE CASTRO & OXILLO

## Group Information
                                           ## Group Name
## Members: De Castro, Rod Andrei & Oxillo, Ivan Russ
## Assigned Theme:
## Topic: Adidad US Sale Datasets

# Project Overview
## Dataset Title: Adidad US Sale Datasets

## Source 
The Adidas Sales Dataset was published on Kaggle by Heemali Chaudhari. It contains sales transaction records for Adidas products, and is available for download on the Kaggle platform (Chaudhari, n.d.). https://www.kaggle.com/datasets/heemalichaudhari/adidas-sales-dataset
    
## License/Attribution:

## Objective:
To analyze the dataset through a thorough exploration of its structure, including the identification of missing values, outliers, and inconsistencies. This process also involves determining the necessary preprocessing steps such as data cleaning, transformation, and preparation for further analysis.

## Implementation Guide (Step-by-Step Process)

### Data Preprocessing
### Step 1. Data Processing - Importing Libraries and Loading Dataset
* Import essential libraries such as **pandas** for data handling and **scikit-learn** modules (`StandardScaler`, `OneHotEncoder`, `SimpleImputer`, `ColumnTransformer`, `Pipeline`) for preprocessing.  
* Load the dataset **(Adidas US Sales Datasets.xlsx)** into a pandas DataFrame for further analysis.

## Step 1: Data Processing – Importing Libraries

Import essential libraries for data manipulation and preprocessing, including **pandas** for handling datasets and **scikit-learn** modules for scaling, encoding, imputing, and building pipelines.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = pd.read_excel('Adidas US Sales Datasets.xlsx')

display(df.head())

df.info()
