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
    test = df_train['OUTCOME'].values

    X_train, X_test, y_train, y_test = model_selection.train_test_split(train, test, stratify=test, test_size=0.2)

    pipe = pipeline.Pipeline([('scaler', preprocessing.StandardScaler()),
                              ('model', ensemble.ExtraTreesClassifier(class_weight='balanced', max_depth=2,
                                                                      min_samples_leaf=0.5, min_samples_split=2))])
    pipe.fit(X_train, y_train)
    preds_proba = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)

    '''params = {'model__max_depth': [2, 3, 5, 7],
              'model__min_samples_split': [1, 2, 3, 5, 7, 9],
              'model__min_samples_leaf': [0.5, 0.7, 0.9, 1.0],
              'model__class_weight': ['balanced'],
              'model__max_leaf_nodes': []}

    grid = model_selection.GridSearchCV(pipe, params, cv=3, verbose=5, n_jobs=-1, scoring='neg_log_loss')
    grid.fit(X_train, y_train)
    print(grid.score(X_test, y_test))
    print(grid.best_params_)'''

    log_loss = metrics.log_loss(y_test, preds_proba)
    print(f'Log Loss: {log_loss}')

    final_preds = pipe.predict_proba(df_test)
    final_preds = final_preds[:, 1]
    submission = pd.DataFrame({'OUTCOME': final_preds})
    submission.to_csv('Submission.csv', index=False)


print(run())
