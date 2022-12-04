import pandas as pd

from sklearn import pipeline
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn import ensemble


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

    categories_train = [col for col in df_train.columns if df_train[col].dtypes == 'object']
    df_train = pd.get_dummies(data=df_train, columns=categories_train)

    categories_test = [col for col in df_test.columns if df_test[col].dtypes == 'object']
    df_test = pd.get_dummies(data=df_test, columns=categories_test)

    '''categories_train = [col for col in df_train.columns if df_train[col].dtypes == 'object']
    for col in categories_train:
        le = preprocessing.LabelEncoder()
        df_train.loc[:, col] = le.fit_transform(df_train[col])

    categories_test = [col for col in df_test.columns if df_test[col].dtypes == 'object']
    for col in categories_test:
        le = preprocessing.LabelEncoder()
        df_test.loc[:, col] = le.fit_transform(df_test[col])'''

    train = df_train.drop(['ID', 'OUTCOME'], axis=1)
    test = df_train['OUTCOME']

    X = train
    y = test
    skf = model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=5)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # X_train, X_test, y_train, y_test = model_selection.train_test_split(train, test, stratify=test, test_size=0.2)

    pipe = pipeline.Pipeline([('scaler', preprocessing.StandardScaler()),
                              ('model', ensemble.ExtraTreesClassifier(random_state=0, n_estimators=100, max_depth=3))])
    pipe.fit(X_train, y_train)
    preds_proba = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)

    auc_roc_score = metrics.roc_auc_score(y_test, preds)
    log_loss = metrics.log_loss(y_test, preds_proba)
    print(f'AUC ROC Score: {auc_roc_score}')
    print(f'Log Loss: {log_loss}')

    final_preds = pipe.predict_proba(df_test)
    final_preds = final_preds[:, 1]
    submission = pd.DataFrame({'OUTCOME': final_preds})
    submission.to_csv('Submission.csv', index=False)


print(run())
