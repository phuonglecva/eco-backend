import numpy


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


def generate_iip_table(values, names):
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
