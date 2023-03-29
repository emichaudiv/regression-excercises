import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from env import user, password, host

import warnings
warnings.filterwarnings('ignore')

def acquire_zillow():
    if os.path.exists('zillow_2017.csv'):
        return pd.read_csv('zillow_2017.csv', index_col=0)
    else:

        url = f"mysql+pymysql://{user}:{password}@{host}/zillow"
        query = """
        SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
        FROM properties_2017
        LEFT JOIN propertylandusetype USING(propertylandusetypeid)
        WHERE propertylandusedesc IN ("Single Family Residential","Inferred Single Family Residential")"""

        df = pd.read_sql(query, url)


        df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                                  'bathroomcnt':'bathrooms', 
                                  'calculatedfinishedsquarefeet':'square_feet',
                                  'taxvaluedollarcnt':'tax_value', 
                                  'yearbuilt':'year_built',})
        df.to_csv("zillow_2017.csv", index=False)
        return df

def remove_outliers(df, k, col_list):
    for col in col_list:
        q1, q3 = df[col].quantile([.25, .75])         
        iqr = q3 - q1         
        upper_bound = q3 + k * iqr   
        lower_bound = q1 - k * iqr   
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df

def get_hist(df):    
    plt.figure(figsize=(16, 3))
    cols = [col for col in df.columns if col not in ['fips', 'year_built']]
    for i, col in enumerate(cols):
        plot_number = i + 1 
        plt.subplot(1, len(cols), plot_number)
        plt.title(col)
        df[col].hist(bins=5)
        plt.grid(False)
        plt.ticklabel_format(useOffset=False)
        plt.tight_layout()
    plt.show()        
        
def get_box(df):
    cols = ['bedrooms', 'bathrooms', 'area', 'tax_value', 'taxamount']
    plt.figure(figsize=(16, 3))
    for i, col in enumerate(cols):
        plot_number = i + 1 
        plt.subplot(1, len(cols), plot_number)
        plt.title(col)
        sns.boxplot(data=df[[col]])
        plt.grid(False)
        plt.tight_layout()
    plt.show()

def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'area', 'tax_value', 'taxamount'])
    get_hist(df)
    get_box(df)
    df.fips = df.fips.astype(object)
    df.year_built = df.year_built.astype(object)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    imputer = SimpleImputer(strategy='median')
    imputer.fit(train[['year_built']])
    train[['year_built']] = imputer.transform(train[['year_built']])
    validate[['year_built']] = imputer.transform(validate[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])   
    return train, validate, test    

def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow(acquire_zillow())
    
    return train, validate, test