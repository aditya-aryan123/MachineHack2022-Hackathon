import pandas as pd

from sklearn import pipeline
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb


def run():
    df_train = pd.read_csv('../input/train.csv')
    df_test = pd.read_csv('../input/test.csv')

    df_train['OUTCOME'] = df_train['OUTCOME'].astype('int64')
    y = df_train.OUTCOME.values

    data = pd.concat((df_train, df_test)).reset_index(drop=True).copy()
    data = data.drop(columns=['ID', 'OUTCOME'], axis=1)

    data['POSTAL_CODE'] = data['POSTAL_CODE'].astype('int64')
    data['VEHICLE_OWNERSHIP'] = data['VEHICLE_OWNERSHIP'].astype('int64')
    data['MARRIED'] = data['MARRIED'].astype('int64')
    data['CHILDREN'] = data['CHILDREN'].astype('int64')

    features = [col for col in data.columns if data[col].dtypes == 'object']
    data = pd.get_dummies(data=data, columns=features)

    train = data[:len(df_train)].values
    test = data[len(df_train):].values

    X_train, X_test, y_train, y_test = model_selection.train_test_split(train, y, test_size=0.25, random_state=42)

    pipe = pipeline.Pipeline([('scaler', preprocessing.StandardScaler()),
                              ('model', xgb.XGBClassifier(eta=0.05, max_depth=3,
                                                          reg_lambda=0, scale_pos_weight=1,
                                                          subsample=0.8, colsample_bytree=0.5,
                                                          min_child_weight=13, reg_alpha=10,
                                                          gamma=0))])
    pipe.fit(X_train, y_train)
    preds_proba = pipe.predict_proba(X_test)[:, 1]

    log_loss = metrics.log_loss(y_test, preds_proba)
    print(f'Log Loss: {log_loss}')

    final_preds = pipe.predict_proba(test)
    final_preds = final_preds[:, 1]
    submission = pd.DataFrame({'OUTCOME': final_preds})
    submission.to_csv('Submission.csv', index=False)


print(run())
