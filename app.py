from re import M
from flask import Flask
from flask import json
from flask.json import jsonify
from flask_restful import Api, Resource
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexing import convert_to_index_sliceable
from data_loader import (
    get_cpi_data,
    get_export_data,
    get_iip_data,
    get_iip_data_by_month_year,
    get_iip_data_from_to,
    get_import_data,
    get_import_data_by_interval,
    get_import_data_by_month_year,
    get_import_data_for_report,
    get_import_month_list_and_year_list,
    get_month_and_year_iip_list,
    get_thuchi_data_by_month,
    get_thuchi_data_fromto,
    get_thuchi_year_list_and_month_list,
    get_unemployment_report_by_year,
    get_unemployment_report_from_to,
    get_unemployment_year_list,
    read_cpi,
    read_iip,
    get_cpi_timeline,
    get_unemployment_rate,
    get_revenue_expenditure,
    load_gdp,
    load_xnk,
)
from flask_cors import CORS, cross_origin
from model.data_loader import *
from flask import request
from datetime import datetime, time
import numpy as np
import joblib
import scipy.stats as st
import requests
from utils import generate_iip_table, generateColumnsByMonth, generateColumnsDataUnemployment, generateColumnsForIip, generateColumnsForRevenue, generateColumnsForUnemployment, generateColumsForImport, generateDataForRevenue, generateImportTable, generateValuesForTables
from datetime import timedelta, timezone

# jwt import 
from flask_jwt_extended import create_access_token
from flask_jwt_extended import get_jwt_identity
from flask_jwt_extended import jwt_required
from flask_jwt_extended import JWTManager
from flask_jwt_extended import create_access_token
from flask_jwt_extended import get_jwt
from flask_jwt_extended import get_jwt_identity
from flask_jwt_extended import jwt_required
from flask_jwt_extended import JWTManager
from flask_jwt_extended import set_access_cookies

# define flask app
app = Flask(__name__, static_folder="build", static_url_path="/")

app.config["JWT_SECRET_KEY"] = "secret"  # Change this!
app.config["JWT_TOKEN_LOCATION"] = ["headers", "cookies", "json", "query_string"]
app.config["JWT_COOKIE_SECURE"] = False
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)

jwt = JWTManager(app)
@app.after_request
def refresh_expiring_jwts(response):
    try:
        exp_timestamp = get_jwt()["exp"]
        now = datetime.now(timezone.utc)
        target_timestamp = datetime.timestamp(now + timedelta(minutes=30))
        if target_timestamp > exp_timestamp:
            access_token = create_access_token(identity=get_jwt_identity())
            set_access_cookies(response, access_token)
        return response
    except (RuntimeError, KeyError):
        # Case where there is not a valid JWT. Just return the original respone
        return response

app.url_map.strict_slashes = False
api = Api(app)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

# load model
cpi_models = joblib.load("saved_model/cpi_forecast_model.joblib")
iip_models = joblib.load("saved_model/iip_forecast_model.joblib")
import_models = joblib.load("saved_model/import_forecast_model.joblib")
export_models = joblib.load("saved_model/export_forecast_model.joblib")

def get_cpi_forecast(models, next=3, alpha=0.95):
    arima_models = models["arima_models"]
    reg_models = models["reg_models"]
    start_time = models["time_test"][0]
    std = models["std_avg"]
    print(f"std: {std}")

    start_m, start_y = start_time.split("/")[0], start_time.split("/")[1]
    if len(start_y) < 4:
        start_y = f"20{start_y}"
    start_time = f"{start_m}/{start_y}"

    # get list timeline
    timeline = pd.date_range(start=start_time, periods=next, freq="M")
    timeline = [datetime.strftime(t, format="%m/%Y") for t in timeline]

    arima_output = [m.predict(next).tolist() for m in arima_models]
    arima_output_arr = np.array(arima_output).T

    reg_out = [
        m.predict(arima_output_arr, alpha=1 - alpha)[0].tolist() for m in reg_models
    ]
    reg_out = np.sum(reg_out, axis=0) / len(reg_models)

    interval_forcasts = [
        m.predict(arima_output_arr, alpha=1 - alpha)[1].tolist() for m in reg_models
    ]
    interval_arr = np.array(interval_forcasts, dtype="float")
    print(interval_arr.shape)
    lower_arr = interval_arr[:, :, 0, 0]
    upper_arr = interval_arr[:, :, 1, 0]

    lower = np.sum(lower_arr, axis=0) / len(reg_models)
    upper = np.sum(upper_arr, axis=0) / len(reg_models)
    return reg_out.tolist(), lower.tolist(), upper.tolist(), timeline


def get_forecast(models, next=3, alpha=0.95):
    t_multiplier = st.t.ppf(alpha + (1 - alpha) / 2, len(models["y_train"]) - 1)
    arima_models = models["arima_models"]
    reg_models = models["reg_models"]
    start_time = models["time_test"][0]
    std = models["std_avg"]
    print(f"std: {std}")

    start_m, start_y = start_time.split("/")[0], start_time.split("/")[1]
    if len(start_y) < 4:
        start_y = f"20{start_y}"
    start_time = f"{start_m}/{start_y}"

    # get list timeline
    timeline = pd.date_range(start=start_time, periods=next, freq="M")
    timeline = [datetime.strftime(t, format="%m/%Y") for t in timeline]

    arima_output = [m.predict(next).tolist() for m in arima_models]
    interval_ouput = [
        m.predict(next, return_conf_int=True)[1].tolist() for m in arima_models
    ]
    interval_arr = np.array(interval_ouput)
    arima_output_arr = np.array(arima_output).T
    arima_ouput_lower_arr = interval_arr[:, :, 0].T
    arima_ouput_upper_arr = interval_arr[:, :, 1].T

    reg_out = [m.predict(arima_output_arr).tolist() for m in reg_models]
    lower_reg_out = [m.predict(arima_ouput_lower_arr).tolist() for m in reg_models]
    upper_reg_out = [m.predict(arima_ouput_upper_arr).tolist() for m in reg_models]

    reg_out = np.sum(reg_out, axis=0) / len(reg_models)
    lower_reg_out = np.sum(lower_reg_out, axis=0) / len(reg_models)
    upper_reg_out = np.sum(upper_reg_out, axis=0) / len(reg_models)

    # count std for predictions (normal distribution)
    # upper1 = reg_out + 1.96 * std
    # lower1 = reg_out - 1.96 * std

    # count std for predictions (student's t distribution)
    upper1 = reg_out + t_multiplier * std
    lower1 = reg_out - t_multiplier * std

    lower = [np.min([x, y]) for x, y in zip(lower1, lower_reg_out)]
    upper = [np.max([x, y]) for x, y in zip(upper1, upper_reg_out)]

    return reg_out.tolist(), lower_reg_out.tolist(), upper_reg_out.tolist(), timeline
    # return reg_out.tolist(), lower, upper, timeline
    # return reg_out.tolist(), lower1.tolist(), upper1.tolist(), timeline

@app.route("/")
def index():
    return app.send_static_file("index.html")
    # return {"data": "helello"}

@app.route("/login")
def login():
    token = request.args.get("token", None)
    url = f'http://sso.ai2c.asia/org/authentication/Authenticate?token={token}'
    req = requests.get(url)
    res = req.json()
    
    isExpire = res["Expired"]
    username = res["Username"]
    if isExpire:
        return jsonify({
            "data": {
                "message": "Token is expired"
            }
        })
    else:
        access_token = create_access_token(identity=username)

        return jsonify({
            "data": {
                # "token": token,
                # "tokenInfo": req.json(),
                "expire": isExpire,
                "accessToken": access_token,
                "message": "Login successfully",
                # "userInfo": req.json()
            }
        })

@app.route("/api/v1/<city>/revenue-report")
@jwt_required()
def get_revenue_report_data(city):
    # print(get_jwt_identity())    
    month = request.args.get("month", None)
    year = request.args.get("year", None)
    from_month = request.args.get("fromMonth", None)
    from_year = request.args.get("fromYear", None)
    to_month = request.args.get("toMonth", None)
    to_year = request.args.get("toYear", None)
    cols = [
            {
                "title": "TT",
                "dataIndex": "tt",
                "key": "tt",
                "width": 100,
                "fixed": "left",
            },
            {
                "title": "Tháng",
                "dataIndex": "index_name",
                "key": "index_name",
                "width": 100,
                "fixed": "left",
            },
            {
                "title": "Đơn vị",
                "dataIndex": "unit",
                "key": "unit",
                "width": 100,
                "fixed": "left",
            },
        ]

    if month and year:
        names, columns, timeline,  values_list = get_thuchi_data_by_month(city, month, year)
    else:
        names, columns, timeline,  values_list = get_thuchi_data_fromto(city, from_month, from_year, to_month, to_year)

    revenue_cols = generateColumnsForRevenue(names[0], columns[0], 0)
    expenditure_cols = generateColumnsForRevenue(names[1], columns[1], len(columns[0]))
    cols.append(revenue_cols)
    cols.append(expenditure_cols)

    revenue_data = generateDataForRevenue(values_list[0], timeline, 0)
    expenditure_data = generateDataForRevenue(values_list[1], timeline, len(columns[0]))
    month_list, year_list = get_thuchi_year_list_and_month_list(city)
    
    columnsData = np.concatenate(values_list, axis=1)
    columnsData = generateDataForRevenue(columnsData, timeline, 0)
    return jsonify({
        "name": names.tolist(),
        # "columnsData": [revenue_data, expenditure_data],
        "columnsData": columnsData,
        "columns": cols,
        "timeline": timeline.tolist(),
        "month_list": month_list,
        "year_list": year_list
    })

@app.route("/api/v1/<city>/unemployment-report")
def get_unemployment_report_data(city):
    # get time params
    year = request.args.get("year", None)
    from_year = request.args.get("fromYear", None)
    to_year = request.args.get("toYear", None)
    columns = [
        {
            "title": "TT",
            "dataIndex": "tt",
            "key": "tt",
            "width": 100,
            "fixed": "left",
        },
        {
            "title": "Tháng",
            "dataIndex": "index_name",
            "key": "index_name",
            "width": 100,
            "fixed": "left",
        },
        {
            "title": "Đơn vị",
            "dataIndex": "unit",
            "key": "unit",
            "width": 100,
            "fixed": "left",
        },
    ]

    if year:
        names, timelines, values = get_unemployment_report_by_year(city, year)
    else:
        names, timelines, values = get_unemployment_report_from_to(city, from_year, to_year)
    
    for i in range(len(names)):
        columns.append(generateColumnsForUnemployment(names[i], i * 9))
    
    columnsData = generateColumnsDataUnemployment(values, timelines)

    year_list = get_unemployment_year_list(city)
    return jsonify({
        "name": names.tolist(),
        "columnsData": columnsData,
        "columns": columns,
        "timeline": timelines.tolist(),
        "month_list": None,
        "year_list": year_list
    })

    


@app.route("/api/v1/<city>/import-report")
def get_import_report_data(city):
    # get time params
    month = request.args.get("month", None)
    year = request.args.get("year", None)
    from_month = request.args.get("fromMonth", None)
    from_year = request.args.get("fromYear", None)
    to_month = request.args.get("toMonth", None)
    to_year = request.args.get("toYear", None)

    columns = [
        {
            "title": "TT",
            "dataIndex": "tt",
            "key": "tt",
            "width": 100,
            "fixed": "left",
        },
        {
            "title": "Chỉ tiêu",
            "dataIndex": "index_name",
            "key": "index_name",
            "width": 100,
            "fixed": "left",
        },
        {
            "title": "Đơn vị",
            "dataIndex": "unit",
            "key": "unit",
            "width": 100,
            "fixed": "left",
        },
    ]

    if month and year:
        name, data, timeline = get_import_data_by_month_year(city, month, year)
    else:
        name, data, timeline = get_import_data_by_interval(city, from_month, from_year, to_month, to_year)    

    for i in range(len(timeline)):
        columns.append(generateColumsForImport(timeline[i], i * 6))
    
    columnsData = generateImportTable(data, name)
    
    month_list, year_list  = get_import_month_list_and_year_list(city)
    return jsonify({
        "name": name.tolist(),
        "columnsData": columnsData,
        "columns": columns,
        "timeline": timeline.tolist(),
        "month_list": month_list,
        "year_list": year_list
    })

@app.route("/api/v1/<city>/export-report")
def get_export_report_data(city):
    # get time params
    month = request.args.get("month", None)
    year = request.args.get("year", None)
    from_month = request.args.get("fromMonth", None)
    from_year = request.args.get("fromYear", None)
    to_month = request.args.get("toMonth", None)
    to_year = request.args.get("toYear", None)

    columns = [
        {
            "title": "TT",
            "dataIndex": "tt",
            "key": "tt",
            "width": 100,
            "fixed": "left",
        },
        {
            "title": "Chỉ tiêu",
            "dataIndex": "index_name",
            "key": "index_name",
            "width": 100,
            "fixed": "left",
        },
        {
            "title": "Đơn vị",
            "dataIndex": "unit",
            "key": "unit",
            "width": 100,
            "fixed": "left",
        },
    ]

    if month and year:
        name, data, timeline = get_import_data_by_month_year(city, month, year, sheet_name=1)
    else:
        name, data, timeline = get_import_data_by_interval(city, from_month, from_year, to_month, to_year, sheet_name=1)    

    for i in range(len(timeline)):
        columns.append(generateColumsForImport(timeline[i], i * 6))
    
    columnsData = generateImportTable(data, name)
    
    month_list, year_list  = get_import_month_list_and_year_list(city, sheet_name=1)
    return jsonify({
        "name": name.tolist(),
        "columnsData": columnsData,
        "columns": columns,
        "timeline": timeline.tolist(),
        "month_list": month_list,
        "year_list": year_list
    })

@app.route("/api/v1/<city>/iip-report")
def get_iip_report_data(city):
    # get time params
    month = request.args.get("month", None)
    year = request.args.get("year", None)
    from_month = request.args.get("fromMonth", None)
    from_year = request.args.get("fromYear", None)
    to_month = request.args.get("toMonth", None)
    to_year = request.args.get("toYear", None)

    month_list, year_list = get_month_and_year_iip_list(city)
    # columns
    columns = [
        {
            "title": "TT",
            "dataIndex": "tt",
            "key": "tt",
            "width": 100,
            "fixed": "left",
        },
        {
            "title": "Chỉ tiêu",
            "dataIndex": "index_name",
            "key": "index_name",
            "width": 100,
            "fixed": "left",
        },
        {
            "title": "Đơn vị",
            "dataIndex": "unit",
            "key": "unit",
            "width": 100,
            "fixed": "left",
        },
    ]
    timeline = []
    data = []
    # columns = []
    columnsData = []
    name = []
    # get list month,
    if month and year:
        data = get_iip_data_by_month_year(city, month, year)
    else:
        data = get_iip_data_from_to(city, from_month, from_year, to_month, to_year)

    timeline = data["timeline"]
    
    name = data["name"]

    data1, data2 = data["data"]
    for i in range(len(timeline)):
        colName = generateColumnsForIip(timeline[i], i * 2)
        columns.append(colName)

    columnsData = generate_iip_table(data1["data"], data2["data"], name)
    return jsonify({
        "timeline": timeline.tolist(),
        "index_name": name.tolist(),
        # "data": data,
        "columns": columns,
        "columnsData": columnsData,
        "year_list": year_list,
        "month_list": month_list
    })

@app.route("/api/v1/<city>/cpi-report")
@cross_origin()
def get_cpi_report_data(city):
    month = request.args.get("month", None)
    year = request.args.get("year", None)

    # columns
    columns = [
        {
            "title": "TT",
            "dataIndex": "tt",
            "key": "tt",
            "width": 100,
            "fixed": "left",
        },
        {
            "title": "Chỉ tiêu",
            "dataIndex": "index_name",
            "key": "index_name",
            "width": 100,
            "fixed": "left",
        },
        {
            "title": "Đơn vị",
            "dataIndex": "unit",
            "key": "unit",
            "width": 100,
            "fixed": "left",
        },
    ]
    # params for from to options
    fromMonth = request.args.get("fromMonth", None)
    fromYear = request.args.get("fromYear", None)
    toMonth = request.args.get("toMonth", None)
    toYear = request.args.get("toYear", None)

    index_names, timeline, index_values = get_cpi_data(city)

    short_time = [t.split(" ")[-1] for t in timeline]

    # get list month,
    month_list = set([int(t.split("/")[0]) for t in short_time[:-2]])
    month_list = list(sorted(month_list))

    # get year list
    year_list = set([int(t.split("/")[-1]) for t in short_time[:-2]])
    year_list = list(sorted(year_list))
    # print(fromMonth, fromYear, toMonth, toYear)
    # print(short_time)
    if fromMonth and fromYear and toMonth and toYear:
        print("ok")
        start_id = -1
        end_id = -1
        for i in range(len(short_time) - 2):
            if short_time[i] == f"{fromMonth}/{fromYear}":
                start_id = i
            if short_time[i] == f"{toMonth}/{toYear}":
                end_id = i

        if start_id == -1:
            return jsonify(
                {
                    {
                        "data": f"Không có dữ liệu cpi tỉnh {city} từ tháng {fromMonth} năm {fromYear}"
                    }
                }
            )
        elif end_id == -1:
            end_id = len(short_time) - 3

        values_by_id = index_values[:, start_id : end_id + 1, :].tolist()
        timeline = timeline[start_id : end_id + 1]
        for i in range(len(timeline)):
            subs_columns = generateColumnsByMonth(timeline[i], i * 5)
            columns.append(subs_columns)
        data = generateValuesForTables(values_by_id, index_names)
        return jsonify(
            {
                "name": index_names,
                "timeline": timeline,
                "value": values_by_id,
                "year_list": year_list,
                "month_list": month_list,
                "columns": columns,
                "data": data
            }
        )
    time_id = -1

    for i in range(len(short_time)):
        if short_time[i] == f"{month}/{year}":
            time_id = i
            break
    if time_id == -1:
        return jsonify(
            {"data": f"Không có dữ liệu cpi tỉnh {city} tháng {month} năm {year}"}
        )

    values_by_id = index_values[:, time_id, :].tolist()
    timeline = timeline[time_id]
    columns.append(generateColumnsByMonth(timeline, 0))
    data = generateValuesForTables(values_by_id, index_names)
    return jsonify(
        {
            "name": index_names,
            "timeline": timeline,
            "value": values_by_id,
            "year_list": year_list,
            "month_list": month_list,
            "columns": columns,
            "data": data
        }
    )


@app.route("/api/v1/<city>/current-status")
def get_current_index_values(city):
    result = []
    # get cpi data
    cpi_data = read_cpi(city)
    cpi_timeline = get_cpi_timeline(city)

    # get iip data
    values, index, iip_timeline = read_iip(city)

    # get unemployment data
    unE_timeline, _, unEmData = get_unemployment_rate(None, False, city)
    # get export data
    export_data = get_export_data(city)
    # get import data
    import_data = get_import_data()

    # get cpi data
    cpi_d = cpi_data[0]["val"][-1]
    cpi_t = cpi_timeline[-1]
    cpi_name = cpi_data[0]["name"]
    result.append({"data": cpi_d, "time": cpi_t, "name": cpi_name})
    # get iip data
    iip_data = values[0]
    iip_name = index[0]
    iip_t = iip_timeline[-1]
    result.append({"data": iip_data, "time": iip_t, "name": iip_name})

    # get export data
    ex_data = export_data["total_export"]["value"][-1]
    ex_name = export_data["total_export"]["name"]
    ex_t = export_data["timeline"][-1]
    ex_t = "-".join(ex_t.split("/")[::-1])
    result.append({"data": ex_data, "time": ex_t, "name": ex_name})

    # get export data
    im_data = import_data["total_import"]["value"][-1]
    im_name = import_data["total_import"]["name"]
    im_t = import_data["timeline"][-1]
    im_t = "-".join(im_t.split("/")[::-1])
    result.append({"data": im_data, "time": im_t, "name": im_name})

    unEm_timeline, _, unEmData = get_unemployment_rate(None, False, city)
    unEmp_t = unEm_timeline[-1]
    unEmpDataTemp = []
    for value in unEmData:
        name = value["index_name"]
        val = value["value"][0]["value"]["chung"][-1]
        unEmpDataTemp.append({"name": name, "val": val})

    result.append({"data": unEmpDataTemp, "time": unEmp_t, "name": "Tỷ lệ thất nghiệp"})

    return jsonify({"data": result})


@app.route("/api/v1/dong-nai/cpis/forecast/<next>")
def cpi(next):
    alpha = request.args.get("alpha", None)
    if alpha:
        out, lower, upper, timeline = get_cpi_forecast(
            cpi_models, int(next), float(alpha)
        )
    else:
        out, lower, upper, timeline = get_cpi_forecast(cpi_models, int(next))
    return {"data": {"cpi": out, "lower": lower, "upper": upper, "timeline": timeline}}


@app.route("/api/v1/dong-nai/iips/forecast/<next>/")
def iip(next):
    alpha = request.args.get("alpha", None)
    if alpha:
        out, lower, upper, timeline = get_forecast(iip_models, int(next), float(alpha))
    else:
        out, lower, upper, timeline = get_forecast(iip_models, int(next))
    return {"data": {"iip": out, "lower": lower, "upper": upper, "timeline": timeline}}


@app.route("/api/v1/dong-nai/import/forecast/<next>")
def import_(next):
    alpha = request.args.get("alpha", None)
    if alpha:
        out, lower, upper, timeline = get_forecast(
            import_models, int(next), float(alpha)
        )
    else:
        out, lower, upper, timeline = get_forecast(import_models, int(next))

    return {
        "data": {"import": out, "lower": lower, "upper": upper, "timeline": timeline}
    }


@app.route("/api/v1/dong-nai/export/forecast/<next>")
def export_(next):
    alpha = request.args.get("alpha", None)
    if alpha:
        out, lower, upper, timeline = get_forecast(
            export_models, int(next), float(alpha)
        )
    else:
        out, lower, upper, timeline = get_forecast(export_models, int(next))
    return {
        "data": {"export": out, "lower": lower, "upper": upper, "timeline": timeline}
    }


# Define route
@app.route("/api/v1/<city>/cpies")
def get_cpi(city):
    cpi_data = read_cpi(city)
    timeline = get_cpi_timeline(city)
    return {"data": {"timeline": timeline, "cpi": cpi_data}}


@app.route("/api/v1/<city>/cpies/<int:num_month>")
def get_cpi_by_num_month(city, num_month):
    reverse = request.args.get("reverse", None) in ["True", "true", "t", 1]
    cpi_data = read_cpi(city)
    timeline = get_cpi_timeline(city)
    return {
        "params": reverse,
        "data": {
            "timeline": timeline[-num_month:][::-1]
            if reverse
            else timeline[-num_month:],
            "cpi": [
                {
                    "name": row["name"],
                    "val": row["val"][-num_month:][::-1]
                    if reverse
                    else row["val"][-num_month:],
                }
                for row in cpi_data
            ],
        },
    }


@app.route("/api/v1/<city>/cpies/forecast_on_train")
def forecast_on_train_data(city):
    forecasts = load_forecast("forecast_")
    timeline = get_cpi_timeline(city)
    return {
        "data": {
            "forecast_on_train": {
                "cpi": forecasts["train_pred"],
                "from_time": timeline[:-3],
            }
        }
    }


@app.route("/api/v1/<city>/iips")
def get_iip(city):
    values, index, timeline = read_iip(city)
    return {
        "data": {
            "timeline": timeline,
            "iip": values[0],
            "subs": [
                {"name": name, "value": val} for name, val in zip(index[1:], values[1:])
            ],
        }
    }


@app.route("/api/v1/<city>/iips/<int:num_month>")
def get_iip_by_num_month(city, num_month):
    reverse = request.args.get("reverse", None) in ["True", "true", "t", 1]
    values, index, timeline = read_iip(city)
    return {
        "data": {
            "timeline": timeline[-num_month:][::-1]
            if reverse
            else timeline[-num_month:],
            "iip": values[0][-num_month:][::-1] if reverse else values[0][-num_month:],
            "subs": [
                {
                    "name": name,
                    "value": val[-num_month:][::-1] if reverse else val[-num_month:],
                }
                for name, val in zip(index[1:], values[1:])
            ],
        }
    }


@app.route("/api/v1/<city>/cpies/forecast/<int:next>")
def forecast_cpi(city, next):
    forecasts = load_forecast()
    index_names = get_cpi_name()
    timeline = get_cpi_timeline(city)
    return {
        "data": {
            "forecasts": {
                "cpi": forecasts["cpi"][:next],
                "subs": [
                    {"name": name, "val": sub[:next]}
                    for name, sub in zip(index_names[1:], forecasts["subs"])
                ],
                "from_time": timeline[-3],
            }
        }
    }


@app.route("/api/v1/<city>/unemployment", strict_slashes=False)
def get_unemployment(city):
    timeline, _, data = get_unemployment_rate(None, False, city)
    return {"data": {"unemployment": data, "timeline": timeline}}


@app.route("/api/v1/<city>/unemployment/<int:nm>")
def get_unemployment_by_nm(city, nm):
    reverse = request.args.get("reverse", None) in ["True", "true", "t", 1]

    timeline, _, data = get_unemployment_rate(nm, reverse, city)
    return {"data": {"unemployment": data, "timeline": timeline}}


@app.route("/api/v1/<city>/thuchi", strict_slashes=False)
def get_thuchi(city):
    data, timeline = get_revenue_expenditure(None, False, city)
    return jsonify({"data": {"thuchi_data": data, "timeline": timeline}})


@app.route("/api/v1/<city>/thuchi/<int:nm>")
def get_thuchi_by_nm(city, nm):
    reverse = request.args.get("reverse", None) in ["True", "true", "t", 1]
    print(reverse, nm)
    data, timeline = get_revenue_expenditure(nm, reverse, city)
    return jsonify({"data": {"thuchi_data": data, "timeline": timeline}})


@app.route("/api/v1/<city>/gdps")
def get_gdp(city):
    data = load_gdp(city)
    return jsonify({"data": data})


@app.route("/api/v1/<city>/gdps/<int:nm>")
def get_gdp_by_nm(city, nm):
    reverse = request.args.get("reverse", None) in ["True", "true", "t", 1]
    data = load_gdp(city)

    return jsonify(
        {
            "data": {
                "year": data["year"][-nm:][::-1] if reverse else data["year"][-nm:],
                "values": data["values"][-nm:][::-1]
                if reverse
                else data["values"][-nm:],
                "rates": data["rates"][-nm:][::-1] if reverse else data["rates"][-nm:],
                "value_unit": "ty dong",
            }
        }
    )


@app.route("/api/v1/<city>/xnk")
def get_xnk(city):
    xk_val, xk, nk_val, nk, years = load_xnk(city)
    names = [
        "Kim ngạch tháng hiện tại",
        "Tính chung từ đầu năm",
        "Kinh tế nhà nước",
        "Kinh tế ngoài nhà nước",
        "Kinh tế có vốn đầu tư nước ngoài",
    ]
    xk_array = np.array(xk, dtype="float").T
    xk_array[np.isnan(xk_array)] = 0
    nk_array = np.array(nk, dtype="float").T
    nk_array[np.isnan(nk_array)] = 0

    xk_val = np.array(xk_val, dtype="float").T
    xk_val[np.isnan(xk_val)] = 0
    nk_val = np.array(nk_val, dtype="float").T
    nk_val[np.isnan(nk_val)] = 0
    years = [year[:-4] + "-" + year[-4:] for year in list(years)]
    return jsonify(
        {
            "data": {
                "xuatkhau": [
                    {"name": name, "val": val, "rate": rate}
                    for name, val, rate in zip(
                        names, xk_val.tolist(), xk_array.tolist()
                    )
                ],
                "nhapkhau": [
                    {"name": name, "val": val, "rate": rate}
                    for name, val, rate in zip(
                        names, nk_val.tolist(), nk_array.tolist()
                    )
                ],
                "years": years,
            }
        }
    )


@app.route("/api/v1/<city>/xnk/<int:num_month>")
def get_xnk_by_num_month(city, num_month):
    reverse = request.args.get("reverse", None) in ["True", "true", "t", 1]
    xk_val, xk, nk_val, nk, years = load_xnk(city)
    names = [
        "Kim ngạch tháng hiện tại",
        "Tính chung từ đầu năm",
        "Kinh tế nhà nước",
        "Kinh tế ngoài nhà nước",
        "Kinh tế có vốn đầu tư nước ngoài",
    ]
    xk_array = np.array(xk, dtype="float").T
    xk_array[np.isnan(xk_array)] = 0
    nk_array = np.array(nk, dtype="float").T
    nk_array[np.isnan(nk_array)] = 0

    xk_val = np.array(xk_val, dtype="float").T
    xk_val[np.isnan(xk_val)] = 0
    nk_val = np.array(nk_val, dtype="float").T
    nk_val[np.isnan(nk_val)] = 0
    years = [year[:-4] + "-" + year[-4:] for year in list(years)]
    return jsonify(
        {
            "data": {
                "xuatkhau": [
                    {
                        "name": name,
                        "val": val[:num_month][::-1] if reverse else val[:num_month],
                        "rate": rate[:num_month][::-1] if reverse else rate[:num_month],
                    }
                    for name, val, rate in zip(
                        names, xk_val.tolist(), xk_array.tolist()
                    )
                ],
                "nhapkhau": [
                    {
                        "name": name,
                        "val": val[:num_month][::-1] if reverse else val[:num_month],
                        "rate": rate[:num_month][::-1] if reverse else rate[:num_month],
                    }
                    for name, val, rate in zip(
                        names, nk_val.tolist(), nk_array.tolist()
                    )
                ],
                "years": years[:num_month][::-1] if reverse else years[:num_month],
            }
        }
    )


@app.route("/api/v1/<city>/import/")
def get_import(city):
    import_data = get_import_data()
    return jsonify({"data": import_data})


@app.route("/api/v1/<city>/export/")
def get_export(city):
    export_data = get_export_data(city)
    return jsonify({"data": export_data})


@app.errorhandler(404)
def handler_404_err(err):
    return jsonify(
        {
            "type": "404",
            "message": "api not found, please check again",
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
