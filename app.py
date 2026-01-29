from flask import Flask, render_template, request
import pandas as pd
import glob
import joblib
import numpy as np
import calendar
from datetime import datetime
from flask import jsonify
import xgboost as xgb

gw_model = xgb.XGBRegressor()
gw_model.load_model("Groundwater_model.json")



wqi_model = xgb.XGBRegressor()
wqi_model.load_model("wqi_xgb_v1.json")
print("MODEL FEATURES:", wqi_model.get_booster().feature_names)



model = joblib.load("model.pkl")
irrigation_model = joblib.load("irrigation_model.pkl")
FEATURE_COLUMNS = joblib.load("feature_columns.pkl")

LIVE_FLOW_DATA = {
    "flow_lpm": 0.0,
    "total_liters": 0.0,
    "timestamp": None,
    "ward": None
}


files = {
    "data/april.csv": 4,
    "data/may.csv": 5,
    "data/june.csv": 6,
    "data/july.csv": 7,
    "data/august.csv": 8,
    "data/september.csv": 9
}

df_list = []

for file, month in files.items():
    df = pd.read_csv(file)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Fix typo if present
    if "Number of Connectionss" in df.columns:
        df.rename(
            columns={"Number of Connectionss": "Number of Connections"},
            inplace=True
        )

    # 🔥 ADD MONTH HERE
    df["Month"] = month
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)

df.columns = df.columns.str.strip()

print("COLUMNS:", df.columns.tolist())


df["consumption_per_connection"] = (
    df["Consumption in ML"] / df["Number of Connections"]
)


# Load irrigation reference data for dropdowns
irrigation_files = [
    "data/TNVillageSchedule.csv",
    "data/UPVillageSchedule.csv"
]

irrigation_df = pd.concat(
    [pd.read_csv(f) for f in irrigation_files],
    ignore_index=True
)

# Clean column names just in case
irrigation_df.columns = irrigation_df.columns.str.strip()


VALID_DISTRICTS = sorted(
    irrigation_df["district_name"]
    .dropna()
    .unique()
)



VALID_WARDS = sorted(df["Ward number"].dropna().unique())
VALID_NAMES = sorted(df["Ward Name"].dropna().unique())

FEATURES = [
    "Ward number",
    "month_sin",
    "month_cos",
    "Number of Connections",
    "consumption_per_connection"
]

print("Loaded csv files:", files)
print("Valid wards:", VALID_WARDS)
print("Valid ward names:", VALID_NAMES)

app = Flask(__name__)

def forecast_ward(ward_number, months_ahead, gowth_rate=0.01):
    ward_data = df[df["Ward number"] == ward_number].sort_values("Month")
    last_row = ward_data.iloc[-1]

    current_month = datetime.now().month
    connections = last_row["Number of Connections"]
    cpc = last_row["consumption_per_connection"]

    future_rows = []

    for i in range(1, months_ahead + 1):
        connections *= (1 + gowth_rate)
        future_rows.append({
            "Ward number": ward_number,
            "month_sin": np.sin(2 * np.pi * (current_month + i) / 12),
            "month_cos": np.cos(2 * np.pi * (current_month + i) / 12),
            "Number of Connections": connections,
            "consumption_per_connection": cpc,
            "Month": current_month + i
        })

    future_df = pd.DataFrame(future_rows)
    future_df["Predicted Consumption"] = model.predict(
        future_df[FEATURES]
    )

    return future_df

def month_number_to_name(month_num):
    return calendar.month_name[(month_num - 1) % 12 + 1]






def predict_irrigation(inputs: dict):
    """
    inputs: dict from form
    returns: predicted net irrigated area + groundwater stress
    """

    # build dataframe
    df_input = pd.DataFrame([inputs])

    # one-hot encode state
    df_input = pd.get_dummies(df_input, columns=["state"])

    # align columns with training
    df_input = df_input.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # predict irrigation
    predicted_irrigation = irrigation_model.predict(df_input)[0]

    # compute groundwater stress (derived, NOT predicted)
    stress = (
        inputs["avg_ground_water_level_pre_monsoon"]
        - inputs["avg_ground_water_level_post_monsoon"]
    )

    return predicted_irrigation, stress




@app.route("/", methods=["GET"])
def home():
    return  render_template("index.html")

 
@app.route("/consumption", methods=["GET"])
def consumption_input():
    ward_map = (
        df[["Ward number", "Ward Name"]]
        .drop_duplicates()
        .set_index("Ward number")["Ward Name"]
        .to_dict()
    )
    reverse_ward_map = {v: k for k, v in ward_map.items()}

    return render_template(
        "consumption_input.html",
        wards=VALID_WARDS,
        names=VALID_NAMES,
        ward_map=ward_map,
        reverse_ward_map=reverse_ward_map
    )

@app.route("/irrigation", methods=["GET"])
def irrigation_input():
    return render_template(
        "irrigation_input.html",
        districts=VALID_DISTRICTS
    )

 
    reverse_ward_map = {v: k for k, v in ward_map.items()}


    return render_template(
        "index.html",
        wards=VALID_WARDS,
        names=VALID_NAMES,
        districts = VALID_DISTRICTS,
        ward_map=ward_map,
        reverse_ward_map=reverse_ward_map
    )

@app.route("/forecast", methods=["POST"])
def forecast():
    region = request.form.get("region", "").strip()
    time = request.form.get("time")

    print("region:", region)
    print("time:", time)
    #return "OK"

    ward = int(region)
    months = int(time)
    ward_name = df.loc[df["Ward number"] == ward, "Ward Name"].iloc[0]


    result = forecast_ward(ward, months, gowth_rate=0.03)

    result["Month Name"] = result["Month"].apply(month_number_to_name)
    total = result["Predicted Consumption"].sum()
    print("Forecast result:\n", result)

    avg_value = result["Predicted Consumption"].mean()

    peak_idx = result["Predicted Consumption"].idxmax()
    peak_month = result.loc[peak_idx, "Month Name"]
    peak_value = result.loc[peak_idx, "Predicted Consumption"]

    feature_importance = {
    "Number of Connections": 0.653292,
    "Consumption per Connection": 0.323205,
    "Ward Number": 0.256674,
    "Month (sin)": 0.072564,
    "Month (cos)": 0.069265
}


    return render_template(
    "result.html",
    ward=ward,
    ward_name=ward_name,
    table=result.to_dict(orient="records"),
    total_ml=round(total, 2),
    avg_value=round(avg_value, 2),
    peak_month=peak_month,
    peak_value=round(peak_value, 2),
    feature_importance=feature_importance
)




@app.route("/irrigation-predict", methods=["POST"])
def irrigation_predict():

    try:
        inputs = {
            "geographical_area": float(request.form["geographical_area"]),
            "cultivable_area": float(request.form["cultivable_area"]),
            "net_sown_area": float(request.form["net_sown_area"]),
            "gross_irrigated_area_kharif_season": float(request.form["gross_irrigated_area_kharif"]),
            "gross_irrigated_area_rabi_season": float(request.form["gross_irrigated_area_rabi"]),
            "gross_irrigated_area_perennial_season": float(request.form["gross_irrigated_area_perennial"]),
            "gross_irrigated_area_other_season": float(request.form["gross_irrigated_area_other"]),
            "avg_ground_water_level_pre_monsoon": float(request.form["gw_pre_monsoon"]),
            "avg_ground_water_level_post_monsoon": float(request.form["gw_post_monsoon"]),
            "state": request.form["state"],
            "district": request.form["district_name"]
        }

    except (KeyError, ValueError):
        return "Invalid irrigation input", 400
    
    feature_importance = {
    "Geographical Area": 0.174004,
    "Irrigation Intensity": 0.137934,
    "Irrigation Coverage": 0.123303,
    "Perennial Irrigation": 0.108244,
    "Kharif Irrigation": 0.100785,
    "Cultivable Area": 0.082018,
    "Rabi Irrigation": 0.078691,
    "Net Irrigated Area": 0.070586,
    "Net Sown Area": 0.067111,
    "Other Season Irrigation": 0.057324
}

    predicted_irrigation, stress = predict_irrigation(inputs)

    return render_template(
        "irrigation_result.html",
        predicted_irrigation=round(predicted_irrigation, 2),
        groundwater_stress=round(stress, 2),
        district=inputs["district"],
        feature_importance=feature_importance
    )


LOGS = []

@app.route("/api/log", methods=["POST"])
def receive_log():
    data = request.json or {}
    msg = data.get("msg", "")
    ward = data.get("ward", "unknown")

    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] Ward {ward}: {msg}"

    LOGS.append(line)

    # keep last 100 logs
    if len(LOGS) > 100:
        LOGS.pop(0)

    print(line)   # 👈 shows in Flask terminal
    return jsonify({"status": "ok"})


@app.route("/logs")
def view_logs():
    return "<br>".join(LOGS[::-1])



@app.route("/api/flow", methods=["POST"])
def receive_flow():
    data = request.json
    

    LIVE_FLOW_DATA["flow_lpm"] = data["flow_lpm"]
    LIVE_FLOW_DATA["total_liters"] = data["total_liters"]
    LIVE_FLOW_DATA["ward"] = data.get("ward")
    LIVE_FLOW_DATA["timestamp"] = datetime.now().isoformat()

    return jsonify({"status": "ok"})


@app.route("/api/flow/live", methods=["GET"])
def get_live_flow():
    return jsonify(LIVE_FLOW_DATA)


@app.route("/live")
def live():
    return render_template("live.html")

MODEL_FEATURES = wqi_model.get_booster().feature_names

WQI_FEATURES = [
    "Potential of Hydrogen (pH)",
    "Dissolved oxygen (mg/L)",
    "Total Dissolved Solids (mg/L)",
    "Nitrate N (mgN/L)",
    "Chloride (mg/L)",
    "Sulphate (mg/L)",
    "Total Hardness (mgCaCO3/L)",
    "Iron(mg/L)",
    "Total Alkalinity (mg/L as CaCO3)",
    "Sodium (mg/L)",
    "year",
    "station_id",
    "month_sin",
    "month_cos"
]


def predict_wqi(inputs):
    month = inputs["month"]

    row = {
        "Potential of Hydrogen (pH)": inputs["ph"],
        "Dissolved oxygen (mg/L)": inputs["do"],
        "Total Dissolved Solids (mg/L)": inputs["tds"],
        "Nitrate N (mgN/L)": inputs["nitrate_n"],
        "Chloride (mg/L)": inputs["chloride"],
        "Sulphate (mg/L)": inputs["sulphate"],
        "Total Hardness (mgCaCO3/L)": inputs["hardness"],
        "Iron(mg/L)": inputs["iron"],
        "Total Alkalinity (mg/L as CaCO3)": inputs["alkalinity"],
        "Sodium (mg/L)": inputs["sodium"],
        "year": inputs["year"],
        "station_id": int(inputs["station_id"]),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12)
    }

    X = pd.DataFrame([row], columns=WQI_FEATURES)

    dmatrix = xgb.DMatrix(
        X,
        feature_names=WQI_FEATURES
    )

    return float(wqi_model.get_booster().predict(dmatrix)[0])

@app.route("/wqi-predict", methods=["POST"])
def wqi_predict():
    inputs = {
        "ph": float(request.form["ph"]),
        "do": float(request.form["do"]),
        "tds": float(request.form["tds"]),
        "nitrate_n": float(request.form["nitrate"]),
        "chloride": float(request.form["chloride"]),
        "sulphate": float(request.form["sulphate"]),
        "hardness": float(request.form["hardness"]),
        "iron": float(request.form["iron"]),
        "alkalinity": float(request.form["alkalinity"]),
        "sodium": float(request.form["sodium"]),
        "month": int(request.form["month"]),
        "year": int(request.form["year"]),
        "station_id": int(request.form["station_id"])
    }

    wqi = predict_wqi(inputs)
    category = classify_wqi(wqi)


    return render_template(
        "wqi_result.html",
        wqi=round(wqi, 2),
        category = category,
        importance = WQI_FEATURE_IMPORTANCE
    )

WQI_FEATURE_IMPORTANCE = [
    {"feature": "Dissolved oxygen (mg/L)", "importance": 16.8},
    {"feature": "Total Alkalinity (mg/L as CaCO3)", "importance": 16.2},
    {"feature": "Total Dissolved Solids (mg/L)", "importance": 14.5},
    {"feature": "Sodium (mg/L)", "importance": 13.9},
    {"feature": "Chloride (mg/L)", "importance": 12.6},
    {"feature": "Potential of Hydrogen (pH)", "importance": 10.8},
    {"feature": "Iron(mg/L)", "importance": 6.1},
    {"feature": "Sulphate (mg/L)", "importance": 3.9},
    {"feature": "Total Hardness (mgCaCO3/L)", "importance": 3.2},
    {"feature": "Nitrate N (mgN/L)", "importance": 1.7},
    {"feature": "year", "importance": 0.9},
    {"feature": "station_id", "importance": 0.6},
    {"feature": "month_cos", "importance": 0.4},
    {"feature": "month_sin", "importance": 0.4}
]



@app.route("/wqi")
def wqi_home():
    return render_template("wqi_input.html",)


def classify_wqi(wqi):
    if wqi <= 25:
        return "Excellent"
    elif wqi <= 50:
        return "Good"
    elif wqi <= 75:
        return "Poor"
    else:
        return "Unfit"

@app.route("/api/stations")
def api_stations():
    df = pd.read_csv("data/station_lookup.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    stations = df.rename(columns={
        "Station": "station_name",
        "District LGD Code": "station_id",
        "District": "district",
        "State": "state"
    })[
        ["station_id", "station_name", "district", "state"]
    ]

    # Drop duplicates (important — many duplicates exist)
    stations = stations.drop_duplicates(subset=["station_id", "station_name"])

    # Sort for UX
    stations = stations.sort_values("station_name")

    return jsonify(stations.to_dict(orient="records"))


GW_FEATURES = [
    'GW_Lag1_SameSeason',
    'GW_Level_Lag1',
    'Well_Depth (meters)',
    'Season',
    'TYPE',
    'SOURCE',
    'Aquifer',
    'Year',
    'State_log'
]


def classify_level(level):
    """
    level: groundwater depth in meters below ground level (mbgl)
    """

    if level < 5:
        return "Very Shallow (Flood / Waterlogging Risk)"
    elif level < 10:
        return "Shallow (Healthy)"
    elif level < 20:
        return "Moderate (Watch)"
    elif level < 40:
        return "Deep (Stressed)"
    else:
        return "Very Deep (Critical)"


@app.route("/groundwater")
def groundwater_home():
    return render_template("groundwater_input.html")



def predict_groundwater(inputs):
    X = pd.DataFrame([inputs], columns=GW_FEATURES)
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        raise ValueError("Invalid groundwater input")

    dmatrix = xgb.DMatrix(X, feature_names=GW_FEATURES)
    return float(gw_model.get_booster().predict(dmatrix)[0])

@app.route("/groundwater-predict", methods=["POST"])
def groundwater_predict():
    inputs = {
        "GW_Lag1_SameSeason": float(request.form["lag_same"]),
        "GW_Level_Lag1": float(request.form["lag"]),
        "Well_Depth (meters)": float(request.form["depth"]),
        "Season": int(request.form["season"]),
        "TYPE": int(request.form["type"]),
        "SOURCE": int(request.form["source"]),
        "Aquifer": int(request.form["aquifer"]),
        "Year": int(request.form["year"]),
        "State_log": np.log1p(int(request.form["state_lgd"]))
    }

    level = predict_groundwater(inputs)

    return render_template(
        "groundwater_result.html",
        level=round(level, 2),
        status=classify_level(level)
    )








    errors = {}

    #  Validation
    if region not in VALID_WARDS:
        errors["region"] = "Invalid ward or district"

    if not time:
        errors["time"] = "Please select a time period"

    if errors:
        return render_template(
            "input.html",
            errors=errors,
            region=region,
            time=time
        )

    #  READY FOR MODEL
    print("=== USER INPUT RECEIVED ===")
    print("Region:", region)
    print("Time (months):", time)
    print("==========================")

    # Placeholder for model output
    return f"Received input for {region}, {time} months"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
