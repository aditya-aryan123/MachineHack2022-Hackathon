# MachineHack2022-Hackathon


## Overview

The Vehicle Insurance business is a multi-billion dollar industry. Every year millions and millions of premiums are paid, and a huge amount of claims also pile up. 

You have to step into the shoes of a data scientist who is building models to help an insurance company understand which claims are worth rejecting and the claims which should be accepted for reimbursement. 

You are given a rich dataset consisting of thousands of rows of past records, which you can use to learn more about your customers’ behaviours. For example, you are supposed to create an ML model to look at a case of an insurance claim and decide whether to reject or accept it. 

Columns: ['ID', 'AGE', 'GENDER', 'DRIVING_EXPERIENCE', 'EDUCATION', 'INCOME',
          'CREDIT_SCORE', 'VEHICLE_OWNERSHIP', 'VEHICLE_YEAR', 'MARRIED',
          'CHILDREN', 'POSTAL_CODE', 'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS',
          'DUIS', 'PAST_ACCIDENTS', 'OUTCOME', 'TYPE_OF_VEHICLE']

 Learn and predict the OUTCOME variable.
 
 
 
 ## Files
 
Analysis.ipynb: Analysis File
 
Untitles1.ipynb: Analysis File

train_ohe_xgb.py: I used One Hot Encoding and used XGBoost Classifier model.
 
oe_te_xgb.py: I used Ordinal Encoding, Target Encoding and build XGBoost Classifier.

train_fe_ohe_xgb.py: I used Frequency Encoding, One Hot Encoding and build XGBoost Classifier.

train_oe_le_xgb.py: I used Ordinal Encoding, Label Encoding and build XGBoost Classifier.

train_ohe_dt.py: I used One Hot Encoding and build Decision Tree Classifier.

train_ohe_etc.py: I used One Hot Encoding and build Extra Tree Classifier.

grid_etc.py: Grid Search on Extra Tree Classifier.


## Leaderboard Ranking

Leaderboard Rank: 106 Out of 1008
