from os import name, set_inheritable
import pandas as pd
import numpy as np
from config import variables
from datetime import datetime, time

data_dir = variables['data_dir']

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


def read_cpi(city, filename='/cpi_data.csv'):
    df = pd.read_csv('%s%s/%s' % (data_dir, city, filename), header=None)
    list_index_name = get_cpi_name()
    data = df.values.tolist()
    return [{'name': name, 'val': val} for name, val in zip(list_index_name, data)]


def read_iip(city, filename='iip_data.xlsx'):
    df = pd.read_excel('%s%s/%s' %(data_dir, city, filename))
    index_list = df.iloc[:, 0]
    columns = df.columns[1:].tolist()
    columns = [t.strftime('%Y-%m') for t in columns]
    values = df.values[:, 1:]
    return values.tolist(), index_list.tolist(), columns


def get_cpi_timeline(city, filename='cpi_timeline.xlsx'):
    df = pd.read_excel('%s%s/%s' %(data_dir, city, filename))
    list_timeline = df.columns.tolist()
    list_timeline = [t.strftime('%Y-%m') for t in list_timeline]
    return list_timeline


def get_unemployment_rate(nm, reverse, city, filename='thatnghiep.xlsx'):

    df = pd.read_excel('%s%s/%s' %(data_dir, city, filename))
    data = df.iloc[2:, :]
    time = data.iloc[:, 0].values
    values = data.iloc[:, 1:].values
    cat = ['Tỷ lệ thất nghiệp từ 15 tuổi trở lên',
           'Tỷ lệ thất nghiệp trong độ tuổi lao động']
    start_index = [0, 9]
    res = []
    regions = ['Tong so', 'Thanh thi', 'Nong thon']
    if not nm:
        nm = len(values[:, 0])
    for cat_name, start in zip(cat, start_index):
        temp_list = []
        for i, region in zip(range(start, start + 9)[0:-1:3], regions):
            temp = {}
            temp['chung'] = values[:, i].tolist()[-nm:][::-1] if reverse else values[:, i].tolist()[-nm:]
            temp['nam'] = values[:, i + 1].tolist()[-nm:][::-1] if reverse else values[:, i + 1].tolist()[-nm:]
            temp['nu'] = values[:, i + 2].tolist()[-nm:][::-1] if reverse else values[:, i + 2].tolist()[-nm:]
            temp_list.append({
                'region': region,
                'value': temp
            })
        res.append({
            'index_name': cat_name,
            'value': temp_list
        })
    years = [int(t) for t in time]
    return years[-nm:][::-1] if reverse else years[-nm:], values, res

def get_revenue_expenditure(nm, reverse, city, filename='thuchingansach.xlsx'):
    df = pd.read_excel('%s%s/%s' %(data_dir, city, filename))
    df.fillna(0,inplace=True)
    data = df.values[1:, :]
    timeline = data[:, 0]
    values = np.nan_to_num(data[:, 1:], nan=0)
    cat = ['Thu ngan sach', 'Chi ngan sach']
    sub_cat = [['Tổng thu (tỷ đồng)',	'Thu nội địa (tỷ đồng)',	'Thu lĩnh vực XNK( tỷ đồng)'],
               ['Tổng chi (tỷ đồng)', '	Chi đầu tư phát triển (tỷ đồng)	',
                'Chi thường xuyên']
               ]
    cat_start_id = [0, 9]
    res = []
    for cat_name, start_id, sub_cat_names in zip(cat, cat_start_id, sub_cat):
        # temp = {}
        temp = []
        for sub_cat_name, id in zip(sub_cat_names, range(start_id, start_id + 9)[0:-1:3]):
            temp_val = np.nan_to_num(values[:, id]).tolist()
            temp.append({
                'cat_name': sub_cat_name.strip(),
                'value': temp_val[:nm][::-1] if reverse else temp_val[:nm]
            })
            # temp[sub_cat_name.strip()] = np.nan_to_num(values[:, id]).tolist()
        res.append({
            'index_name': cat_name.strip(),
            'value': temp
        })
        # res[cat_name.strip()] = temp
    years = (pd.to_datetime(timeline).strftime('%m-%Y').values.tolist())
    return res, years[:nm][::-1] if reverse else years[:nm]
    # return  res, timeline.tolist()

def load_gdp(city, filename='gdp.xlsx'):
    data = pd.read_excel('%s%s/%s' %(data_dir, city, filename))
    year, values, rates = data.values[1:,0], data.values[1:, 1], data.values[1:, 2]
    values = [float(str(val)
                    .replace('.', '?')
                    .replace(',', '.')
                    .replace('?', ''))
              for val in values
              ]
    rates = [float(str(r).replace(',', '.')) for r in rates]
    return {
        'year': year.tolist(),
        'values': values,
        'rates': rates,
        'value_unit': 'ty dong'
    }

def load_xnk(city, filename='xnk.xlsx'):
    years = pd.read_excel('%s%s/%s' %(data_dir, city, filename), None).keys()
    data_list = [pd.read_excel('%s%s/%s' %(data_dir, city, filename), sheet_name=year, header=None) for  year in years]
    xnk_names = ['Xuất khẩu', 'Nhập khẩu']
    xuatkhau_start_id = [data.index[data.values[:, 0] == xnk_names[0]].tolist()[0] for data in data_list]
    nhapkhau_start_id = [data.index[data.values[:, 0] == xnk_names[1]].tolist()[0] for data in data_list]
    xk_data = []
    nk_data = []
    for data, start_id in zip(data_list, xuatkhau_start_id):
        xk_data.append(data.iloc[list(range(start_id, start_id + 7)), [2, 4]].values[-5:, :].T.tolist())
    for data, start_id in zip(data_list, nhapkhau_start_id):
        nk_data.append(data.iloc[list(range(start_id, start_id + 6)), [2, 4]].values[-5:, :].T.tolist())
    
    xk_data_value  = [[str(x).replace('.', '').replace(',', '.') for x in xk[0]] for xk in xk_data]
    xk_data_rate  = [[str(x).replace('%', '').replace(',','.').replace(';','').strip() for x in xk[1]] for xk in xk_data]
    nk_data_value  = [[str(x).replace('.', '').replace(',', '.') for x in xk[0]] for xk in nk_data]
    nk_data_rate  = [[str(x).replace('%', '').replace(',','.').replace(';','').strip() for x in xk[1]] for xk in nk_data]
    
    # return xuatkhau_start_id, nhapkhau_start_id, xk_data, nk_data
    return xk_data_value, xk_data_rate, nk_data_value, nk_data_rate, years

def get_export_data(city, filename='thuongmai.xlsx'):
    filename = f'{data_dir}{city}/{filename}'
    total_index = 7
    domestic_index = 8
    foreign_index = 9
    name_index = 1
    goods_list_index = list(range(11, 24))
    export_ = pd.read_excel(filename, header=None)

    export_d_name = export_.iloc[total_index, name_index]
    export_d = export_.iloc[total_index, 4::6][:-2]
    
    domestic_export_d_name = export_.iloc[domestic_index, name_index]
    domestic_export_d = export_.iloc[domestic_index, 4::6][:-2]
    
    foreign_export_d_name = export_.iloc[foreign_index, name_index]
    foreign_export_d = export_.iloc[foreign_index, 4::6][:-2]
    

    goods_d_name = export_.iloc[goods_list_index, name_index].tolist()
    goods_d = export_.iloc[goods_list_index, 4::6].iloc[:, :-2]

    # convert export data to 'float' type
    export_d = np.array([str(s).replace(',', 'x').replace('.', '').replace(
        'x', '.') for s in export_d.values], dtype='float')
    domestic_export_d = np.array([str(s).replace(',', 'x').replace('.', '').replace(
        'x', '.') for s in domestic_export_d.values], dtype='float')
    foreign_export_d = np.array([str(s).replace(',', 'x').replace('.', '').replace(
        'x', '.') for s in foreign_export_d.values], dtype='float')
    goods_d = np.array([[str(s).replace(',', 'x').replace('.', '').replace(
        'x', '.') for s in row] for row in goods_d.values], dtype='float')
    # get timeline
    timeline = export_.iloc[1, :].values
    timeline = [t.split(' ')[-1] for t in timeline if str(t) != 'nan'][:-2]
    return {
        "total_export": {
            'name': export_d_name,
            'value': export_d.tolist()
        },
        "domestic_export": {
            "name": domestic_export_d_name,
            "value": domestic_export_d.tolist()
        },
        "foreign_export": {
            "name": foreign_export_d_name,
            "value": foreign_export_d.tolist()
        },
        "details": [{"name": name ,"value": value.tolist()} for (name, value) in zip(goods_d_name, goods_d)],
        "timeline": timeline
    }

def get_import_data(filename='data/dong-nai/xuatnhapkhau.xlsx'):
    data = pd.read_excel(filename, sheet_name=0)
    timeline = data.iloc[0, :].values
    timeline = [t.split(' ')[-1] for  t in timeline if str(t) != 'nan'][3:-2]

    main_index = [6]
    domestic_index = [7]
    foreign_index = [8]
    name_index = 1
    sub_indices = list(range(10, 28))

    main_data_name = data.iloc[main_index, name_index].values.tolist()[0]
    main_data = data.iloc[main_index, 4::6]
    
    domestic_data_name = data.iloc[domestic_index, name_index].values.tolist()[0]
    domestic_data = data.iloc[domestic_index, 4::6]
    
    foreign_data_name = data.iloc[foreign_index, name_index].values.tolist()[0]
    foreign_data = data.iloc[foreign_index, 4::6]


    sub_data_name = data.iloc[sub_indices, name_index].tolist()
    sub_data = data.iloc[sub_indices, 4::6]

    main_data_values = np.squeeze(main_data.values[:, 3:-2])
    domestic_data_values = np.squeeze(domestic_data.values[:, 3:-2])
    foreign_data_values = np.squeeze(foreign_data.values[:, 3:-2])

    sub_data_values = sub_data.values[:, 3:-2]

    main_data_values = np.array([str(num).replace(',', 'x').replace('.', '').replace('x', '.') for num in main_data_values], dtype='float')
    domestic_data_values = np.array([str(num).replace(',', 'x').replace('.', '').replace('x', '.') for num in domestic_data_values], dtype='float')
    foreign_data_values = np.array([str(num).replace(',', 'x').replace('.', '').replace('x', '.') for num in foreign_data_values], dtype='float')
    
    
    sub_data_values = np.array([[str(num).replace(',', 'x').replace('.', '').replace('x', '.') for num in row] for row in sub_data_values], dtype='float')

    return {
        "total_import": {
            'name': main_data_name,
            'value': main_data_values.tolist()
        },
        "domestic_import": {
            'name': domestic_data_name,
            'value': domestic_data_values.tolist()
        },
        "foreign_import": {
            'name': foreign_data_name,
            'value': foreign_data_values.tolist()
        },
        "details": [{"name": name ,"value": value.tolist()} for (name, value) in zip(sub_data_name, sub_data_values)],
        "timeline": timeline
    }

def get_cpi_data(city, filename='thuongmai.xlsx'):
    filename = f'{data_dir}{city}/{filename}'
    data = pd.read_excel(filename, sheet_name=-1)
    data.fillna('', inplace=True)
    cpi_data = data.iloc[4:, :]
    timeline = data.iloc[0, 2::5]
    index_names = cpi_data.iloc[:, 1]
    index_values = cpi_data.iloc[:, 2: ]

    index_values = index_values.values.reshape((index_values.shape[0], timeline.shape[0], -1))
    return index_names.tolist(), timeline.tolist(), index_values


def get_iip_data(city, filename='iip_data_.xlsx'):
    filename = f'{data_dir}{city}/{filename}'
    iip_data = pd.read_excel(filename)

    timeline = iip_data.iloc[0, 2::2].values[:-2]
    index_name = iip_data.iloc[3:, 1].values

    colName1 = iip_data.iloc[1, 2]
    colName2 = iip_data.iloc[1, 3]

    col1Data = iip_data.iloc[3:, 2::2].values[:, :-2]
    col2Data = iip_data.iloc[3:, 3::2].values[:, :-2]

    return index_name, timeline, {"name": colName1, "data": col1Data}, {"name": colName2, "data": col2Data}


def get_iip_data_by_month_year(city, month, year):
    name, timeline, data1, data2 = get_iip_data(city)
    short_time = [t.split(" ")[-1] for t in timeline]

    time_id = -1
    for i in range(len(short_time)):
        if short_time[i] == f'{month}/{year}':
            time_id = i
            break
    # Check if iip at month year exist in database
    if time_id == -1:
        return None
    
    ti = timeline[time_id: time_id + 1]
    data1['data'] = data1['data'][:, time_id: time_id + 1]
    data2['data'] = data2['data'][:, time_id: time_id + 1]

    return {
        "timeline": ti,
        "name": name,
        "data": [
            data1, data2
        ]
    }

def get_iip_data_from_to(city, from_month, from_year, to_month, to_year):
    name, timeline, data1, data2 = get_iip_data(city)
    short_time = [t.split(" ")[-1] for t in timeline]

    start_id = -1
    end_id = -1

    for i in range(len(short_time)):
        if short_time[i] == f'{from_month}/{from_year}':
            start_id = i
        if short_time[i] == f'{to_month}/{to_year}':
            end_id = i
    
    if start_id == -1 and start_id == -1:
        return None

    if start_id == -1:
        start_id = 0
    
    if end_id == -1:
        end_id = len(short_time) - 1
    

    ti = timeline[start_id: end_id + 1]
    data1['data'] = data1['data'][:, start_id: end_id + 1]
    data2['data'] = data2['data'][:, start_id: end_id + 1]

    return {
        "timeline": ti,
        "name": name,
        "data": [
            data1, data2
        ]
    }

def get_month_and_year_iip_list(city, filename='iip_data_.xlsx'):
    filename = f'{data_dir}{city}/{filename}'
    iip_data = pd.read_excel(filename)

    timeline = iip_data.iloc[0, 2::2].values[:-2]
    short_time = [t.split(" ")[-1] for t in timeline]

    # get list month,
    month_list = set([int(t.split("/")[0]) for t in short_time[:-2]])
    month_list = list(sorted(month_list))

    # get year list
    year_list = set([int(t.split("/")[-1]) for t in short_time[:-2]])
    year_list = list(sorted(year_list))

    return month_list, year_list


def get_import_data_for_report(city, filename='xuatnhapkhau.xlsx', sheet_name=0):

    filename = f'{data_dir}{city}/{filename}'

    import_data = pd.read_excel(filename, sheet_name=sheet_name)
    import_data.fillna(0, inplace=True)
    timeline = import_data.iloc[0, 3::6].values[:-2]
    time_len = len(timeline)
    index_name = import_data.iloc[6:, 1].values
    values = import_data.iloc[6:, 3:].values[:, :time_len * 6]
    # split values 
    splited_values = np.array_split(values, time_len, axis=1)

    return index_name, splited_values, timeline

def get_import_data_by_month_year(city, month, year, sheet_name=0):
    name, data, timeline = get_import_data_for_report(city, sheet_name=sheet_name)
    shortTime = [t.split(" ")[-1] for t in timeline]
    print(shortTime, f'{month}/{year}')
    timeId = -1
    for i in range(len(shortTime)):
        if shortTime[i] == f'{month}/{year}':
            print("im here")
            timeId = i
            break
    
    if timeId == -1:
        return None
    
    return name, data[timeId: timeId+1], timeline[timeId: timeId + 1]

def get_import_data_by_interval(city, fromMonth, fromYear, toMonth, toYear, sheet_name=0):
    name, data, timeline = get_import_data_for_report(city, sheet_name=sheet_name)
    shortTime = [t.split(" ")[-1] for t in timeline]

    fromId = -1
    toId = -1
    for i in range(len(shortTime)):
        if shortTime[i] == f'{fromMonth}/{fromYear}':
            fromId = i
        if shortTime[i] == f'{toMonth}/{toYear}':
            toId = i
    if fromId == -1 and toId == -1:
        return None
    
    fromId = 0 if fromId == -1 else fromId
    toId = len(shortTime) - 1 if toId == -1 else toId

    return name, data[fromId: toId + 1], timeline[fromId: toId + 1]

def get_import_month_list_and_year_list(city, filename='xuatnhapkhau.xlsx', sheet_name=0):
    filename = f'{data_dir}{city}/{filename}'
    import_data = pd.read_excel(filename, sheet_name=sheet_name)
    timeline = import_data.iloc[0, 3::6].values[:-2]

    short_time = [t.split(" ")[-1] for t in timeline]

    # get list month,
    month_list = set([int(t.split("/")[0]) for t in short_time[:-2]])
    month_list = list(sorted(month_list))

    # get year list
    year_list = set([int(t.split("/")[-1]) for t in short_time[:-2]])
    year_list = list(sorted(year_list))

    return month_list, year_list

def get_unemployment_report_data(city,filename='thatnghiep.xlsx'):
    filename = f'{data_dir}{city}/{filename}'
    data = pd.read_excel(filename)

    names = data.columns.values[1::9]
    timeline = data.iloc[2:, 0]
    values = data.iloc[2:, 1:]

    return names, timeline.values, values.values
    
def get_unemployment_report_by_year(city, year):
    names, timeline, values = get_unemployment_report_data(city)
    print(timeline, year)
    timeId = -1
    for i in range(len(timeline)):
        if int(timeline[i]) == int(year):
            timeId = i
            break
    if timeId == -1:
        return None
    timeline = timeline[timeId: timeId + 1]
    values = values[ timeId: timeId + 1,:]
    return names, timeline, values
    
def get_unemployment_report_from_to(city, fromYear, toYear):
    names, timeline, values = get_unemployment_report_data(city)

    fromId = -1
    toId = -1
    for i in range(len(timeline)):
        if int(timeline[i]) == int(fromYear):
            fromId = i
        if int(timeline[i]) == int(toYear):
            toId = i
    if fromId == -1 and toId == -1: 
        return None
    fromId = fromId if fromId != -1 else 0
    toId = toId if toId != -1 else len(timeline) - 1

    timeline = timeline[fromId: toId + 1]
    values = values[ fromId: toId + 1, :]
    return names, timeline, values

def get_unemployment_year_list(city,filename='thatnghiep.xlsx'):
    filename = f'{data_dir}{city}/{filename}'
    data = pd.read_excel(filename)
    timeline = sorted(set(t for t in data.iloc[2:, 0]))

    return list(timeline)

def get_thuchi_data(city,filename='thuchingansach.xlsx'):
    filename = f'{data_dir}{city}/{filename}'
    data = pd.read_excel(filename)
    columns = np.split(data.iloc[0, :].values[1:-1], 2)

    # generate full timeline 
    timeline = data.iloc[1:, 0].values
    timeline = pd.date_range(timeline[-1], timeline[0], freq='m')
    timeline = pd.Series(timeline).dt.strftime("%m/%y")
    timeline = pd.DataFrame(timeline, columns=["Tháng"])

    data['Tháng'] = data['Tháng'].dt.strftime("%m/%y")

    full_data = pd.merge(timeline, data, how="left", on=["Tháng"])
    full_data.fillna(0, inplace=True)
    # get timeline, values, columns title
    timeline = full_data['Tháng'].values
    values = full_data.values[:, 1:-1]
    values_list = np.split(values, 2, axis=1)
    names = full_data.columns[1::9].values[:-1]

    return names, columns, timeline, values_list

def get_thuchi_data_by_month(city, month, year):
    names, columns, timeline, values_list = get_thuchi_data(city)

    timeId = -1
    for i in range(len(timeline)):
        ti = timeline[i]
        currM, currY = ti.split("/")
        print(currM, currY)
        if int(currM) == int(month) and int(year) == int(currY):
            timeId = i
            break
    
    if timeId == -1:
        return None
    
    return names, columns, timeline[timeId: timeId + 1], [values[timeId: timeId + 1, :] for values in values_list]

def get_thuchi_data_fromto(city, fromMonth, fromYear, toMonth, toYear):
    names, columns, timeline, values_list = get_thuchi_data(city)

    fromId = -1
    toId = -1
    for i in range(len(timeline)):
        ti = timeline[i]
        currM, currY = ti.split("/")
        if int(currM) == int(fromMonth) and int(fromYear) == int(currY):
            fromId = i
        if int(currM) == int(toMonth) and int(toYear) == int(currY):
            toId = i

    if fromId == -1 and toId == -1:
        return None
    fromId = 0 if fromId == -1 else fromId
    toId = len(timeline) - 1 if toId == -1 else toId
    return names, columns, timeline[fromId: toId + 1], [values[fromId: toId + 1, :] for values in values_list]


def get_thuchi_year_list_and_month_list(city,filename='thuchingansach.xlsx'):
    filename = f'{data_dir}{city}/{filename}'
    data = pd.read_excel(filename)

    # generate full timeline 
    timeline = data.iloc[1:, 0].values
    timeline = pd.date_range(timeline[-1], timeline[0], freq='m')
    month_list = pd.Series(timeline).dt.strftime("%m")
    year_list = pd.Series(timeline).dt.strftime("%y")

    month_list = list(sorted(set([m for m in month_list])))
    year_list = list(sorted(set([y for y in year_list])))

    return month_list, year_list