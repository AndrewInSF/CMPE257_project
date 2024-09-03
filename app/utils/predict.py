import joblib
import pandas as pd
from healthcheck.settings import BASE_DIR


def predict(data):
    modelData = {'Heart_rate':int(data["heartRate"]), 
            'Resp_rate':int(data["respRate"]),
            'Body_temp':float(data["bodyTemp"]),
            'Oxy_sat':float(data["oxySat"]),
            'Sys_blood_press':int(data["sysBloodPress"]),
            'Dia_blood_press':int(data["diagPress"]),
            'Age':int(data["age"]),
            'Gender':int(data["gender"]),
            'Weight':float(data["weight"]),
            'Height':float(data["height"]),
            'HRV':float(data["HRV"]),
            'Pulse_press':int(data["pulsePress"]),
            'BMI':float(data["BMI"]),
            'MAP':float(data["map"])
            }

    classifier = joblib.load(BASE_DIR / f"app/prediction_models/best_model.pkl")
        
    current_df = pd.DataFrame([modelData])

    current_predict = classifier.predict(current_df)

    return current_predict[0]

