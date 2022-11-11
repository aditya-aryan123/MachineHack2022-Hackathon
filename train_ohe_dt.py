import pandas as pd

from sklearn.cluster import KMeans
from sklearn import pipeline
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn import tree


def run():
    df_train = pd.read_csv('../input/train.csv')
    df_test = pd.read_csv('../input/test.csv')

    df_train['OUTCOME'] = df_train['OUTCOME'].astype('int64')

    y = df_train.OUTCOME.values

    data = pd.concat((df_train, df_test)).reset_index(drop=True).copy()
    data = data.drop(columns='OUTCOME', axis=1)

    data['HP'] = data['TYPE_OF_VEHICLE'].apply(
        lambda x: 1 if x == 'Sports Car' else 2 if x == 'SUV' else 3 if x == 'Sedan' else 4)
    data['is_YOUNG'] = data['AGE'].apply(lambda x: 1 if x in '16-25' else 0)
    data['is_SENIOR_CITIZEN'] = data['AGE'].apply(lambda x: 1 if x in '65+' else 0)

    data['POSTAL_CODE'] = data['POSTAL_CODE'].astype('int64')
    data['VEHICLE_OWNERSHIP'] = data['VEHICLE_OWNERSHIP'].astype('int64')
    data['MARRIED'] = data['MARRIED'].astype('int64')
    data['CHILDREN'] = data['CHILDREN'].astype('int64')

    categories = [col for col in data.columns if data[col].dtypes == 'object']
    data = pd.get_dummies(data=data, columns=categories)

    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(data)

    kmeans_model = KMeans(5)
    kmeans_model.fit_predict(scaled_df)
    data = pd.concat([data, pd.DataFrame({'cluster': kmeans_model.labels_})], axis=1)

    train = data[:len(df_train)].values
    test = data[len(df_train):].values

    oversample = SMOTE(random_state=42)
    pipe = Pipeline([('o', oversample)])
    X = train
    y = y
    X, y = pipe.fit_resample(X, y)

    skf = model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=5)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    pipe = pipeline.Pipeline([('scaler', preprocessing.StandardScaler()),
                              ('model', tree.DecisionTreeClassifier())])
    pipe.fit(X_train, y_train)
    preds_proba = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)

    log_loss_train = metrics.log_loss(y_train, preds_proba)
    log_loss_test = metrics.log_loss(y_test, preds_proba)
    print(f'Log Loss Train: {log_loss_train}')
    print(f'Log Loss Test: {log_loss_test}')

    final_preds = pipe.predict_proba(test)
    final_preds = final_preds[:, 1]
    submission = pd.DataFrame({'OUTCOME': final_preds})
    submission.to_csv('Submission.csv', index=False)


print(run())
