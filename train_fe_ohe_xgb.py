import pandas as pd

from sklearn import pipeline
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb


def run():
    df_train = pd.read_csv('../input/train.csv')
    df_test = pd.read_csv('../input/test.csv')

    df_test = df_test.drop('ID', axis=1)

    df_train['OUTCOME'] = df_train['OUTCOME'].astype('int64')
    df_train['POSTAL_CODE'] = df_train['POSTAL_CODE'].astype('int64')
    df_train['VEHICLE_OWNERSHIP'] = df_train['VEHICLE_OWNERSHIP'].astype('int64')
    df_train['MARRIED'] = df_train['MARRIED'].astype('int64')
    df_train['CHILDREN'] = df_train['CHILDREN'].astype('int64')

    df_test['POSTAL_CODE'] = df_test['POSTAL_CODE'].astype('int64')
    df_test['VEHICLE_OWNERSHIP'] = df_test['VEHICLE_OWNERSHIP'].astype('int64')
    df_test['MARRIED'] = df_test['MARRIED'].astype('int64')
    df_test['CHILDREN'] = df_test['CHILDREN'].astype('int64')

    df_train['is_YOUNG'] = df_train['AGE'].apply(lambda x: 1 if x in '16-25' else 0)
    df_train['is_SENIOR_CITIZEN'] = df_train['AGE'].apply(lambda x: 1 if x in '65+' else 0)

    df_test['is_YOUNG'] = df_test['AGE'].apply(lambda x: 1 if x in '16-25' else 0)
    df_test['is_SENIOR_CITIZEN'] = df_test['AGE'].apply(lambda x: 1 if x in '65+' else 0)

    features = ['GENDER', 'EDUCATION', 'VEHICLE_YEAR']
    categories = [col for col in df_train.columns if df_train[col].dtypes == 'object' and col not in features]
    encoder_dict = {}
    for var in categories:
        encoder_dict[var] = (df_train[var].value_counts() / len(df_train)).to_dict()
    for var in categories:
        df_train[var] = df_train[var].map(encoder_dict[var])
    df_train = pd.get_dummies(data=df_train, columns=features)

    features = ['GENDER', 'EDUCATION', 'VEHICLE_YEAR']
    categories = [col for col in df_test.columns if df_test[col].dtypes == 'object' and col not in features]
    encoder_dict = {}
    for var in categories:
        encoder_dict[var] = (df_test[var].value_counts() / len(df_test)).to_dict()
    for var in categories:
        df_test[var] = df_test[var].map(encoder_dict[var])
    df_test = pd.get_dummies(data=df_test, columns=features)

    X = df_train.drop(['ID', 'OUTCOME'], axis=1)
    y = df_train['OUTCOME']

    skf = model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=5)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    pipe = pipeline.Pipeline([('scaler', preprocessing.StandardScaler()),
                              ('model', xgb.XGBClassifier(eta=0.4, max_depth=2))])
    pipe.fit(X_train, y_train)
    preds_proba = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)

    log_loss_train = metrics.log_loss(y_train, preds_proba)
    log_loss_test = metrics.log_loss(y_test, preds_proba)
    print(f'Log Loss Train: {log_loss_train}')
    print(f'Log Loss Test: {log_loss_test}')


print(run())
