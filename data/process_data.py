import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    """
    load 2 data base and return merged dataframe
    
    input: 
    messages_filepath- path of message.csv
    categories_filepath- of od category csv
    
    return:
     new df which cotian the 2 input dataframe
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,left_on='id',right_on='id')
    return df



def clean_data(df):
    """
    the function clean the df:
    1.creating diffrent category on the data frame
    2.remove duplicate value
    3. manipulate and create new df that we can work with it to build the ML
    input- row dataframe
    output - clean dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";",expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(x))
    # drop the original categories column from `df`
    df.drop(axis=1,columns=['categories'],inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df=pd.concat([df,categories], axis=1)
    
    # drop duplicates
    df=df.drop_duplicates()
    
    # drop 2
    # Remove rows with a  value of 2 from df
    for i in df.columns:
        df = df[df[i] != 2]
    return  df




def save_data(df, database_filename):
    """
    save the dataframe to future using
    input 
    df - clean dataframe
    database_filename- the desire saved name
    output- NA
    """
    engine = create_engine('sqlite:///ETL_Preparation.db')
    df.to_sql('data_disaster', engine, index=False, if_exists='replace')
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
