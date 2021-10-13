import numpy
import numpy as np

from data_loader import get_iip_data_by_month_year, get_import_data_for_report

def generateColumnsForRevenue(index_name, columns, start_id):
    cols = {
        "title": index_name,
        "width": 100,
        "children": [{
                "title": columns[i],
                "dataIndex": f"name_{start_id + i}",
                "key": f"name_{start_id + i}",
                "width": 100,
            
        } for i in range((len(columns)))]
    }
    return cols
    
def generateDataForRevenue(values, timeline, start_id):
    data = []
    for i in range(len(timeline)):
        tempData = {
            "tt": i + 1,
            "index_name": timeline[i], 
            "unit": "%" if i != 2 else ''
        }
        currData = values[i]
        for j in range(len(currData)):
            tempData[f'name_{start_id + j}'] = currData[j]
        data.append(tempData)

    return data    

def generateColumnsDataUnemployment(values, year_list):
    data = []
    for i  in range(len(year_list)):
        
        tempData = {
            "tt": i + 1,
            "index_name": year_list[i], 
            "unit": "%" if i != 2 else ''
        }
        
        yearData = values[i]
        for j in range(len(yearData)):
            tempData[f'name_{j}'] = yearData[j]

        data.append(tempData)
    return data

def generateColumnsForUnemployment(index_name, start_id):
    type = ["Tổng số", "Thành thị", "Nông thôn"]
    detailType = ["Chung", "Nam", "Nữ"]

    columns = {
        "title": index_name,
        "width":100,
        "children": []
    }
    subsChild = []
    for typeName in type:
        subsChild.append({
            "title": typeName,
            "width": 100,
            "children": [{
                "title": detailType[i],
                "dataIndex": f"name_{start_id + i}",
                "key": f"name_{start_id + i}",
                "width": 100,
            } for i in range(3)]
        })
        start_id += 3
    columns["children"] = subsChild
    return columns

def generateImportTable(values, names):
    data  =[]
    # value shape = ()
    values = np.concatenate(values, axis=1)
    for i in range(len(names)):
        name = names[i]
        rowData = values[i]
        tempData = {
            "tt": i + 1,
            "index_name": name, 
            "unit": "%" if i != 2 else ''
        }
        for j in range(len(rowData)):
            tempData[f'name_{j}'] = rowData[j]
        data.append(tempData)

    return data

def generateColumsForImport(month, start_id):
    index_name = ["Thực hiện tháng trước", "Uớc tính tháng báo cáo", "Cộng dồn từ đầu năm đến cuối tháng báo cáo"]
    print(index_name)
    col_name = ["Số lượng", "Trị giá"]
    topChild = []
    for name in index_name:
        child = []
        child.append({
                "title": col_name[0],
                "dataIndex": f"name_{start_id}",
                "key": f"name_{start_id}",
                "width": 100,
        })
        child.append({
                "title": col_name[1],
                "dataIndex": f"name_{start_id + 1}",
                "key": f"name_{start_id + 1}",
                "width": 100,
        })
        start_id += 2
        topChild.append({
                "title": name,
                "width": 100,
                "children": child
        })

    return {
        "title": month,
        "width": 100,
        "children": topChild
    }


def generateColumnsForIip(month, start_id):
    return {
        "title": month,
        "width": 100,
        "children": [
            {
                "title": "Tháng báo cáo so với tháng trước của năm báo cáo",
                "dataIndex": f"name_{start_id}",
                "key": f"name_{start_id}",
                "width": 100,
            },
            {
                "title": "Tháng báo cáo so với tháng cùng kỳ năm trước",
                "dataIndex": f"name_{start_id + 1}",
                "key": f"name_{start_id + 1}",
                "width": 100,
            },
        ],
    }


def generate_iip_table(data1, data2, names):
    data = []
    for i in range(len(names)):
        name = names[i]

        dataByName = {"tt": i + 1, "index_name": name, "unit": "%" if i != 2 else ""}
        tempData1 = data1[i]
        tempData2 = data2[i]
        tempData = []
        for x, y in zip(tempData1, tempData2):
            tempData.append(x)
            tempData.append(y)
        
        for j in range(len(tempData)):
            dataByName[f'name_{j}'] = tempData[j]

        data.append(dataByName)

    return data


def generateColumnsByMonth(month, start_id):
    topIndex = "Chỉ số giá tháng báo cáo so với (%)"
    index_title = [
        "Kỳ gốc 2014",
        "Cùng tháng năm trước",
        "Tháng 12 năm trước",
        "Tháng trước",
        "Bình quân cùng kỳ",
    ]
    subs = [
        {
            "title": index_title[0],
            "dataIndex": f"name_{start_id}",
            "key": f"name_{start_id}",
            "width": 100,
        }
    ]

    subs_child = []
    for i in range(1, 4):
        start_id += 1
        subs_child.append(
            {
                "title": index_title[i],
                "dataIndex": f"name_{start_id}",
                "key": f"name_{start_id}",
                "width": 100,
            }
        )

    subs.append({"title": topIndex, "width": 100, "children": subs_child})

    start_id += 1
    subs.append(
        {
            "title": index_title[-1],
            "dataIndex": f"name_{start_id}",
            "key": f"name_{start_id}",
            "width": 100,
        }
    )
    return {"title": month, "children": subs, "width": 100}


def generateValuesForTables(values, names):
    data = []
    for i in range(len(names)):
        name = names[i]

        dataByName = {"tt": i + 1, "index_name": name, "unit": "%" if i != 2 else ""}
        value = values[i]
        value_arr = numpy.array(value)
        print(value_arr, value_arr.shape)

        value_arr = value_arr.ravel()

        for j in range((len(value_arr))):
            dataByName[f"name_{j}"] = value_arr[j]

        data.append(dataByName)

    return data



# test
if __name__ == "__main__":
    name, data, timeline = get_import_data_for_report("dong-nai")
    report = generateImportTable(data, name)
    print(report)
