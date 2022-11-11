import pandas as pd
import numpy as np
from sklearn import pipeline
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb


def run():
    df_train = pd.read_csv('../input/train.csv')
    df_test = pd.read_csv('../input/test.csv')

    train = df_train.drop('ID', axis=1)
    test = df_test.drop('ID', axis=1)

    train['OUTCOME'] = train['OUTCOME'].astype('int64')
    train['POSTAL_CODE'] = train['POSTAL_CODE'].astype('int64')
    train['VEHICLE_OWNERSHIP'] = train['VEHICLE_OWNERSHIP'].astype('int64')
    train['MARRIED'] = train['MARRIED'].astype('int64')
    train['CHILDREN'] = train['CHILDREN'].astype('int64')
    train['PAST_ACCIDENTS'] = train['PAST_ACCIDENTS'].astype('int64')
    train['ANNUAL_MILEAGE'] = train['ANNUAL_MILEAGE'].astype('int64')

    test['POSTAL_CODE'] = test['POSTAL_CODE'].astype('int64')
    test['VEHICLE_OWNERSHIP'] = test['VEHICLE_OWNERSHIP'].astype('int64')
    test['MARRIED'] = test['MARRIED'].astype('int64')
    test['CHILDREN'] = test['CHILDREN'].astype('int64')
    test['PAST_ACCIDENTS'] = test['PAST_ACCIDENTS'].astype('int64')
    test['ANNUAL_MILEAGE'] = test['ANNUAL_MILEAGE'].astype('int64')

    train['PAST_ACCIDENTS_BIN'] = pd.cut(train['PAST_ACCIDENTS'], [-1, 1, 4, 16])
    train['DUIS_BIN'] = pd.cut(train['DUIS'], [-1, 1, 3, 7])
    train['SPEEDING_VIOLATIONS_BIN'] = pd.cut(train['SPEEDING_VIOLATIONS'], [-1, 1, 4, 21])
    train['ANNUAL_MILEAGE_BIN'] = pd.cut(train['ANNUAL_MILEAGE'], [1000, 9000, 11000, 13000, 22000])
    train['CREDIT_SCORE_BIN'] = pd.cut(train['CREDIT_SCORE'], [-1, 0.25, 0.5, 0.75, 1.1])

    train['CREDIT_SCORE_BIN'] = train['CREDIT_SCORE_BIN'].astype('object')
    train['PAST_ACCIDENTS_BIN'] = train['PAST_ACCIDENTS_BIN'].astype('object')
    train['DUIS_BIN'] = train['DUIS_BIN'].astype('object')
    train['SPEEDING_VIOLATIONS_BIN'] = train['SPEEDING_VIOLATIONS_BIN'].astype('object')
    train['ANNUAL_MILEAGE_BIN'] = train['ANNUAL_MILEAGE_BIN'].astype('object')

    train = train.drop(['CREDIT_SCORE', 'PAST_ACCIDENTS', 'DUIS', 'SPEEDING_VIOLATIONS', 'ANNUAL_MILEAGE'],
                       axis=1)

    test['PAST_ACCIDENTS_BIN'] = pd.cut(test['PAST_ACCIDENTS'], [-1, 1, 4, 16])
    test['DUIS_BIN'] = pd.cut(test['DUIS'], [-1, 1, 3, 7])
    test['SPEEDING_VIOLATIONS_BIN'] = pd.cut(test['SPEEDING_VIOLATIONS'], [-1, 1, 4, 21])
    test['ANNUAL_MILEAGE_BIN'] = pd.cut(test['ANNUAL_MILEAGE'], [1000, 9000, 11000, 13000, 22000])
    test['CREDIT_SCORE_BIN'] = pd.cut(test['CREDIT_SCORE'], [-1, 0.25, 0.5, 0.75, 1.1])

    test['CREDIT_SCORE_BIN'] = test['CREDIT_SCORE_BIN'].astype('object')
    test['PAST_ACCIDENTS_BIN'] = test['PAST_ACCIDENTS_BIN'].astype('object')
    test['DUIS_BIN'] = test['DUIS_BIN'].astype('object')
    test['SPEEDING_VIOLATIONS_BIN'] = test['SPEEDING_VIOLATIONS_BIN'].astype('object')
    test['ANNUAL_MILEAGE_BIN'] = test['ANNUAL_MILEAGE_BIN'].astype('object')

    test = test.drop(['CREDIT_SCORE', 'PAST_ACCIDENTS', 'DUIS', 'SPEEDING_VIOLATIONS', 'ANNUAL_MILEAGE'],
                     axis=1)

    postal_code_sum = pd.DataFrame(train.groupby('POSTAL_CODE', as_index=True)['OUTCOME'].sum())
    postal_code_count = pd.DataFrame(train.groupby(['POSTAL_CODE'], as_index=True)['OUTCOME'].count())
    combinded_postal_code = pd.merge(postal_code_sum, postal_code_count, how='inner', on=['POSTAL_CODE'])
    combinded_postal_code['POSTAL_CODE_PROPORTION'] = combinded_postal_code['OUTCOME_x'] / combinded_postal_code[
        'OUTCOME_y']
    train = pd.merge(combinded_postal_code, train, how='right', on=['POSTAL_CODE'])
    test = pd.merge(combinded_postal_code, test, how='right', on=['POSTAL_CODE'])
    test['POSTAL_CODE_PROPORTION'].fillna(0.5, inplace=True)

    customer_group_sum = pd.pivot_table(data=train, values='OUTCOME',
                                        index=['AGE', 'GENDER', 'MARRIED', 'CHILDREN', 'EDUCATION', 'INCOME',
                                               'CREDIT_SCORE_BIN', 'DRIVING_EXPERIENCE'],
                                        aggfunc=np.sum)
    customer_group_sum_df = pd.DataFrame(customer_group_sum.to_records())
    customer_group_count = pd.pivot_table(data=train, values='OUTCOME',
                                          index=['AGE', 'GENDER', 'MARRIED', 'CHILDREN', 'EDUCATION', 'INCOME',
                                                 'CREDIT_SCORE_BIN', 'DRIVING_EXPERIENCE'],
                                          aggfunc='count')
    customer_group_count_df = pd.DataFrame(customer_group_count.to_records())
    customer_group_merged = pd.merge(customer_group_sum_df, customer_group_count_df, how='inner',
                                     on=['AGE', 'GENDER', 'MARRIED', 'CHILDREN', 'EDUCATION', 'INCOME',
                                         'CREDIT_SCORE_BIN', 'DRIVING_EXPERIENCE'])
    customer_group_merged['Customer_Acceptance_Rate'] = customer_group_merged['OUTCOME_x'] / customer_group_merged[
        'OUTCOME_y']
    customer_group_merged.drop(['OUTCOME_x', 'OUTCOME_y'], axis=1, inplace=True)
    train = pd.merge(customer_group_merged, train, how='inner',
                     on=['AGE', 'GENDER', 'MARRIED', 'CHILDREN', 'EDUCATION', 'INCOME',
                         'CREDIT_SCORE_BIN', 'DRIVING_EXPERIENCE'])
    test = pd.merge(customer_group_merged, test, how='right',
                    on=['AGE', 'GENDER', 'MARRIED', 'CHILDREN', 'EDUCATION', 'INCOME',
                        'CREDIT_SCORE_BIN', 'DRIVING_EXPERIENCE'])
    test['Customer_Acceptance_Rate'] = test['Customer_Acceptance_Rate'].fillna(0.5)

    vehicle_group_sum = pd.pivot_table(data=train, values='OUTCOME',
                                       index=['VEHICLE_YEAR', 'TYPE_OF_VEHICLE', 'ANNUAL_MILEAGE_BIN',
                                              'SPEEDING_VIOLATIONS_BIN', 'DUIS_BIN', 'PAST_ACCIDENTS_BIN'],
                                       aggfunc=np.sum)
    vehicle_group_sum_df = pd.DataFrame(vehicle_group_sum.to_records())
    vehicle_group_count = pd.pivot_table(data=train, values='OUTCOME',
                                         index=['VEHICLE_YEAR', 'TYPE_OF_VEHICLE', 'ANNUAL_MILEAGE_BIN',
                                                'SPEEDING_VIOLATIONS_BIN', 'DUIS_BIN', 'PAST_ACCIDENTS_BIN'],
                                         aggfunc='count')
    vehicle_group_count_df = pd.DataFrame(vehicle_group_count.to_records())
    vehicle_group_merged = pd.merge(vehicle_group_sum_df, vehicle_group_count_df, how='inner',
                                    on=['VEHICLE_YEAR', 'TYPE_OF_VEHICLE', 'ANNUAL_MILEAGE_BIN',
                                        'SPEEDING_VIOLATIONS_BIN', 'DUIS_BIN', 'PAST_ACCIDENTS_BIN'])
    vehicle_group_merged['Vehicle_Acceptance_Rate'] = vehicle_group_merged['OUTCOME_x'] / vehicle_group_merged[
        'OUTCOME_y']
    vehicle_group_merged.drop(['OUTCOME_x', 'OUTCOME_y'], axis=1, inplace=True)
    train = pd.merge(vehicle_group_merged, train, how='inner',
                     on=['VEHICLE_YEAR', 'TYPE_OF_VEHICLE', 'ANNUAL_MILEAGE_BIN',
                         'SPEEDING_VIOLATIONS_BIN', 'DUIS_BIN', 'PAST_ACCIDENTS_BIN'])
    test = pd.merge(vehicle_group_merged, test, how='right',
                    on=['VEHICLE_YEAR', 'TYPE_OF_VEHICLE', 'ANNUAL_MILEAGE_BIN',
                        'SPEEDING_VIOLATIONS_BIN', 'DUIS_BIN', 'PAST_ACCIDENTS_BIN'])
    test['Vehicle_Acceptance_Rate'] = test['Vehicle_Acceptance_Rate'].fillna(0.5)

    train['PAST_ACCIDENTS_BIN'] = train['PAST_ACCIDENTS_BIN'].astype('object')
    test['PAST_ACCIDENTS_BIN'] = test['PAST_ACCIDENTS_BIN'].astype('object')
    test['ANNUAL_MILEAGE_BIN'] = test['ANNUAL_MILEAGE_BIN'].astype('object')
    test['DUIS_BIN'] = test['DUIS_BIN'].astype('object')

    features = [col for col in train.columns if train[col].dtypes == 'object']
    train = pd.get_dummies(data=train, columns=features)

    features = [col for col in test.columns if test[col].dtypes == 'object']
    test = pd.get_dummies(data=test, columns=features)

    dataframe = pd.read_csv('cleaned_df.csv')
    dataframe20 = pd.read_csv('cleaned_df_20.csv')

    X = dataframe.drop(['Unnamed: 0', 'OUTCOME'], axis=1)
    print(X.columns)
    y = dataframe['OUTCOME']

    '''test = test[['Vehicle_Acceptance_Rate', 'MARRIED', 'CHILDREN', 'Customer_Acceptance_Rate', 'POSTAL_CODE',
                 'POSTAL_CODE_PROPORTION', 'VEHICLE_OWNERSHIP', 'VEHICLE_YEAR_before 2015', 'TYPE_OF_VEHICLE_SUV',
                 'TYPE_OF_VEHICLE_Sedan', 'TYPE_OF_VEHICLE_Sports Car', 'ANNUAL_MILEAGE_BIN_(1000.0, 9000.0]',
                 'AGE_26-39', 'AGE_40-64', 'GENDER_female', 'GENDER_male', 'EDUCATION_none', 'INCOME_poverty',
                 'CREDIT_SCORE_BIN_(0.75, 1.1]', 'DRIVING_EXPERIENCE_10-19y']]'''

    test = test[['Vehicle_Acceptance_Rate', 'MARRIED', 'CHILDREN',
                 'Customer_Acceptance_Rate', 'POSTAL_CODE', 'POSTAL_CODE_PROPORTION',
                 'VEHICLE_OWNERSHIP', 'VEHICLE_YEAR_before 2015',
                 'TYPE_OF_VEHICLE_Sedan', 'TYPE_OF_VEHICLE_Sports Car',
                 'ANNUAL_MILEAGE_BIN_(9000.0, 11000.0]', 'ANNUAL_MILEAGE_BIN_(13000.0, 22000.0]',
                 'SPEEDING_VIOLATIONS_BIN_(-1.0, 1.0]', 'SPEEDING_VIOLATIONS_BIN_(1.0, 4.0]']]

    skf = model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=5)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    pipe = pipeline.Pipeline([('scaler', preprocessing.StandardScaler()),
                              ('model', xgb.XGBClassifier(eta=0.7, max_depth=5,
                                                          reg_lambda=1, scale_pos_weight=1,
                                                          subsample=0.8, colsample_bytree=0.9,
                                                          min_child_weight=9, reg_alpha=10,
                                                          gamma=10))])

    pipe.fit(X_train, y_train)
    preds_proba = pipe.predict_proba(X_test)[:, 1]

    log_loss = metrics.log_loss(y_test, preds_proba)
    print(f'Log Loss: {log_loss}')

    final_preds = pipe.predict_proba(test)
    final_preds = final_preds[:, 1]
    submission = pd.DataFrame({'OUTCOME': final_preds})
    submission.to_csv('Submission.csv', index=False)


print(run())
