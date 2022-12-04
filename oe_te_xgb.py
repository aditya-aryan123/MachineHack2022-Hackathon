import pandas as pd
import category_encoders as ce
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

    columns = ['AGE', 'DRIVING_EXPERIENCE']
    for col in columns:
        oe = preprocessing.OrdinalEncoder()
        df_train.loc[:, col] = oe.fit_transform(df_train[col].values.reshape(-1, 1))
        df_test.loc[:, col] = oe.fit_transform(df_test[col].values.reshape(-1, 1))

    categories = [col for col in df_train.columns if df_train[col].dtypes == 'object']
    for col in categories:
        te = ce.MEstimateEncoder(cols=categories, m=5.0)
        df_train.loc[:, col] = te.fit_transform(df_train[col], y=df_train['OUTCOME'])
        df_test.loc[:, col] = te.fit_transform(df_test[col], y=df_train['OUTCOME'])

    train = df_train.drop(['ID', 'OUTCOME'], axis=1)
    test = df_train['OUTCOME']

    X = train
    y = test
    skf = model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=5)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    pipe = pipeline.Pipeline([('scaler', preprocessing.StandardScaler()),
                              ('model', xgb.XGBClassifier(eta=0.4, max_depth=2))])
    pipe.fit(X_train, y_train)
    preds_proba = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)

    auc_roc_score = metrics.roc_auc_score(y_test, preds)
    log_loss = metrics.log_loss(y_test, preds_proba)
    print(f'AUC ROC Score: {auc_roc_score}')
    print(f'Log Loss: {log_loss}')

    '''final_preds = pipe.predict_proba(test)
    final_preds = final_preds[:, 1]
    submission = pd.DataFrame({'OUTCOME': final_preds})
    submission.to_csv('Submission.csv', index=False)'''


print(run())
