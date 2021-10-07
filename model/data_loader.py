import pandas as pd
import numpy as np


def get_cpi_name():
    name_list = ["Chỉ số giá tiêu dùng chung",
                 "Hàng ăn và dịch vụ ăn uống",
                 "Lương thực",
                 "Thực phẩm",
                 "Ăn uống ngoài gia đình",
                 "Đồ uống và thuốc lá",
                 "May mặc, mũ nón, giáy dép",
                 "Nhà ở, điện, nước, chất đốt, VLXD",
                 "Thiết bị và đồ dùng gia đình",
                 "Thuốc và dịch vụ y tế",
                 "Giao thông",
                 "Bưu chính viễn thông",
                 "Giáo dục",
                 "Văn hóa, giải trí và du lịch",
                 "Hàng hóa và dịch vụ khác"]
    return name_list


def read_cpi_m(filename='model/cpi_data.csv'):
    df = pd.read_csv(filename, header=None)
    list_index_name = get_cpi_name()
    data = df.values.tolist()
    return [{'name': name, 'val': val} for name, val in zip(list_index_name, data)]


def get_sub_cpies():
    data = read_cpi_m()
    main_cpi = data[0]['val']
    subs = [x['val'] for x in data[1:]]
    return main_cpi, subs



def load_forecast(filename='forecast'):
    with open(f'model/data/{filename}.json', 'r') as f:
        import json
        forecasts = json.load(f)

    return forecasts
