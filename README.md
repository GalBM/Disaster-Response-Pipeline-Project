# Disaster Response Pipeline Project

## Motiviation
in the next project I create an we program using ML which classifiy disaster events text to emergency events
furthermore, I experience using ML and data analysis pipeline

## Installation 
he used libraries are: 
pandas
re
sys
json
sklearn
nltk
sqlalchemy
pickle
Flask
plotly
sqlite3

## File description
app - run the web app
data - include all the data which necessry + process_data.py - data 	analysis pipeline
model - include the train_classifier ML pipeline and the classifer


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py` 

4. Click the `PREVIEW` button to open the homepage


##Acknowledgements

I'd like to thank udacity about the tuition and support
