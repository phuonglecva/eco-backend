# import sys
# sys.path.append('.model')
from sklearn import linear_model
# from .data_loader import get_sub_cpies
from model.data_loader import get_sub_cpies
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, ElasticNet


def split_data(vals, nobs=3):
    return vals[:-nobs], vals[-nobs:]


def select_model(train, test):
    min = 1000
    best_model, best_ijk, err = None, None, None
    for i in range(5):
        for j in range(3):
            for k in range(5):
                try:
                    model = ARIMA(train, order=(i, j, k))
                    model_fitted = model.fit()
                    test_hat = model_fitted.forecast(len(test))
                    err = mean_squared_error(test, test_hat[0], squared=False)
                    if err < min:
                        min = err
                        best_model = model_fitted
                        best_ijk = (i, j, k)
                except:
                    continue
    return best_model, best_ijk, min


def cal_loss(model, test):
    test_hat = model.forecast(len(test))[0]
    return abs(test_hat - test) / [test] * 100


def iter_cpi_data(cpi_data):
    res = {}
    for i in range(1):
        train, test = split_data(cpi_data[:(len(cpi_data) - i)])
        model, _, _ = select_model(train, test)
        loss = cal_loss(model, test)
        res[i] = loss
    return res


def train_and_save_models(sub_cpies):
    model_list4sub_cpies = []
    # train phase
    for sub in sub_cpies:
        train, test = split_data(sub)
        model_, _, _ = select_model(train, test)
        model_list4sub_cpies.append(model_)

    # save models
    for i in range(len(model_list4sub_cpies)):
        with open('model/pretrain_model/model_%s' % (i), 'wb') as f:
            pickle.dump(model_list4sub_cpies[i], f)
    return model_list4sub_cpies


def load_models():
    models = []
    for i in range(14):
        with open('model/pretrain_model/model_{}'.format(i), 'rb') as f:
            model = pickle.load(f)
            models.append(model)
    return models


def create_linear_model():
    cpi, subs = get_sub_cpies()
    cpi_data = np.array(cpi)
    subs = np.array(subs, dtype='float')
    cpi_data, cpi_test = cpi_data[:-3], cpi_data[-3:]
    subs = subs[:, :-3].T

    # trainX, testX, trainY, testY = train_test_split(
    #     subs, cpi_data, test_size=0.1)
    model = ElasticNet()
    model.fit(subs, cpi_data)
    train_predict = model.predict(subs)
    return model, train_predict


def forecast_cpi(next=1):
    models = load_models()
    # lingreg = create_linear_model()

    # inputs = [model.forecast(next)[0].tolist() for model in models]
    # inputs = np.array(inputs, dtype='float').T
    # out = lingreg.predict(inputs).tolist()
    return models[0].forcast(next)[0].tolist()


def write_forecast_to_file(next=1):
    models = load_models()
    lingreg, train_pred = create_linear_model()

    inputs = [model.forecast(next)[0].tolist() for model in models]
    inputs = np.array(inputs, dtype='float').T
    out = lingreg.predict(inputs).tolist()
    data = {
        'subs': inputs.T.tolist(),
        'cpi': out,
        "train_pred": train_pred.tolist()
    }
    with open('model/data/forecast_.json', 'w') as f:
        import json
        json.dump(data, f, indent=2)
