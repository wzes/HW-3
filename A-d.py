import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt, floor
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt


def get_distance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    dis = 2 * asin(sqrt(a)) * 6371 * 1000
    return dis


def get_gongcan_map(gongcanDf):
    tmpMap = {}
    for i in range(len(gongcanDf)):
        key = gongcanDf.iloc[i]['RNCID'] + '_' + gongcanDf.iloc[i]['CellID']
        value = [gongcanDf.iloc[i]['Longitude'], gongcanDf.iloc[i]['Latitude']]
        tmpMap[key] = value
    return tmpMap


def process_data(dataDf):
    rnc_ids = dataDf.groupby(['RNCID_1', 'CellID_1'])
    rnc_id_nos = list(rnc_ids.indices.keys())
    datas = []
    data_labels = []
    # Statistic the each point
    for index in range(len(rnc_ids)):
        key = rnc_id_nos[index][0] + '_' + rnc_id_nos[index][1]
        base_longitude = gongcanMap[key][0]
        base_latitude = gongcanMap[key][1]

        rncDf = rnc_ids.get_group(rnc_id_nos[index])
        labels = []
        data = np.zeros((len(rncDf), 25))
        for j in range(len(rncDf)):
            lon = rncDf.iloc[j]['Longitude']
            lat = rncDf.iloc[j]['Latitude']
            label = [int((lon - base_longitude) * pow(10, 20)), int((lat - base_latitude) * pow(10, 20)), lon, lat]
            labels.append(label)
            num_connected = rncDf.iloc[j]['Num_connected']
            tmp = [rncDf.iloc[j]['IMSI'], rncDf.iloc[j]['MRTime']] + [base_longitude, base_latitude]
            for k in range(num_connected):
                location = gongcanMap.get(
                    rncDf.iloc[j]['RNCID_' + str(k + 1)] + '_' + rncDf.iloc[j]['CellID_' + str(k + 1)])
                if location is not None:
                    tmp += [location[0], location[1], rncDf.iloc[j]['RSSI_' + str(k + 1)]]
            for item in range(len(tmp)):
                data[j][item] = tmp[item]
        datas.append(data)
        data_labels.append(np.array(labels))
    return datas, data_labels


def make_picture(errors, labels, color):
    plt.figure(figsize=(15, 8))
    for index in range(len(errors)):
        x = range(len(errors[index]))
        plt.plot(x, errors[index], label='group_' + str(labels[index]), linewidth=0.5, color=color, marker='o',
                 markerfacecolor='blue', markersize=1)
    plt.xlabel('number')
    plt.ylabel('average error')
    plt.title('Average error probability distribution diagram')
    plt.legend()
    plt.show()


def train(classifier, data, label, top_data=None, top_data_label=None):
    distances = []
    for number in range(10):
        # split data
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=number)
        if top_data is not None and top_data_label is not None:
            X_train = np.vstack((X_train, top_data))
            y_train = np.vstack((y_train, top_data_label))
        classifier.fit(X_train[:, 4:], y_train[:, 0:2])
        y_pred = classifier.predict(X_test[:, 4:])
        locations = []
        for index in range(len(y_pred)):
            locations.append([X_test[index][2] + y_pred[index][0] / pow(10, 20),
                              X_test[index][3] + y_pred[index][1] / pow(10, 20)])
        tmp = []
        for index in range(len(y_pred)):
            tmp.append(get_distance(y_test[index][2], y_test[index][3],
                                    locations[index][0],
                                    locations[index][1]))
        distances.append(sorted(tmp))
    errors = []
    matrix = np.array(distances)
    for index in range(matrix.shape[1]):
        errors.append(np.mean(matrix[:, index]))
    return errors


def merge_data(group_medians, datas, data_labels):
    sorted_group_median = sorted(group_medians.items(), key=lambda d: d[1])
    merged_datas = []
    merged_data_labels = []
    k = int(len(datas) / 2)
    top_k_data = datas[sorted_group_median[0][0]]
    top_k_label = data_labels[sorted_group_median[0][0]]
    for index in range(k):
        merged_datas.append(datas[sorted_group_median[index][0]])
        merged_data_labels.append(data_labels[sorted_group_median[index][0]])
        # merge
        if index > 0:
            top_k_data = np.vstack((datas[sorted_group_median[index][0]], top_k_data))
            top_k_label = np.vstack((data_labels[sorted_group_median[index][0]], top_k_label))
    for index in range(k, len(datas)):
        # merge_data = np.vstack((datas[sorted_group_median[index][0]], top_k_data))
        # merge_data_label = np.vstack((data_labels[sorted_group_median[index][0]], top_k_label))
        merged_datas.append(datas[sorted_group_median[index][0]])
        merged_data_labels.append(data_labels[sorted_group_median[index][0]])
    return merged_datas, merged_data_labels, top_k_data, top_k_label, k


if __name__ == "__main__":
    # read data
    dataDf = pd.read_csv('data_2g.csv', header=0, dtype={'RNCID_1': np.object, 'CellID_1': np.object,
                                                         'RNCID_2': np.object, 'CellID_2': np.object,
                                                         'RNCID_3': np.object, 'CellID_3': np.object,
                                                         'RNCID_4': np.object, 'CellID_4': np.object,
                                                         'RNCID_5': np.object, 'CellID_5': np.object,
                                                         'RNCID_6': np.object, 'CellID_6': np.object,
                                                         'RNCID_7': np.object, 'CellID_7': np.object})
    gongcanDf = pd.read_csv('2g_gongcan.csv', header=0, dtype={'RNCID': np.object, 'CellID': np.object})
    # get gongcan data
    gongcanMap = get_gongcan_map(gongcanDf)

    size = len(dataDf)
    # Get the matrix
    minLatitude = dataDf['Latitude'].min()
    maxLatitude = dataDf['Latitude'].max()
    minLongitude = dataDf['Longitude'].min()
    maxLongitude = dataDf['Longitude'].max()

    datas, data_labels = process_data(dataDf)
    average_errors = []
    group_medians = {}
    for index in range(len(datas)):
        rfc = RandomForestClassifier()
        average_error = train(rfc, datas[index], data_labels[index])
        median = np.median(average_error)
        group_medians[index] = median
        average_errors.append(average_error)

    # top k data
    merged_datas, merged_data_labels, top_k_data, top_k_label, k = merge_data(group_medians, datas, data_labels)

    average_errors = []
    for index in range(len(merged_datas)):
        rfc = RandomForestClassifier(max_depth=2, random_state=0)
        if index >= k:
            average_error = train(rfc, merged_datas[index], merged_data_labels[index],
                                  top_data=top_k_data, top_data_label=top_k_label)
        else:
            average_error = train(rfc, merged_datas[index], merged_data_labels[index])
        average_errors.append(average_error)
    color = 'r'
    make_picture(average_errors, range(len(merged_datas)), color)