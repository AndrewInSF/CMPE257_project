import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import joblib
from healthcheck.settings import BASE_DIR
from sklearn.model_selection import train_test_split

def create_model():
    print("create model")
    dataUrl = "human_vital_2024_cleaned.csv"

    df = pd.read_csv(dataUrl)
    print(df.head())

    Y = df['Risk_category'] 
    X = df.drop(columns= ['Risk_category']) 

    idx = list(range(X.shape[0])) 
    train_idx, valid_idx = train_test_split(idx, test_size=0.3, random_state=2021)
    print(">>> # of Train data : {}".format(len(train_idx)))
    print(">>> # of valid data : {}".format(len(valid_idx)))

    best_model = LGBMClassifier(n_estimators=50, learning_rate=0.1, 
                           max_depth=5, reg_alpha=0.5, objective='cross_entropy', 
                           random_state=119)
    best_model.fit(X.iloc[train_idx], Y.iloc[train_idx])

    joblib.dump(best_model, BASE_DIR / f"app/prediction_models/best_model.pkl")


    classifier = joblib.load(BASE_DIR / f"app/prediction_models/best_model.pkl")


    # Train Acc
    y_pre_train = classifier.predict(X.iloc[0:1])
    print(y_pre_train)

    # Test Acc
    y_pre_test = classifier.predict(X.iloc[0:1])
    print(y_pre_test)
    return True