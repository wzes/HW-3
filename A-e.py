import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt, floor
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
    base_locations = []
    # Statistic the each point
    for index in range(len(rnc_ids)):
        key = rnc_id_nos[index][0] + '_' + rnc_id_nos[index][1]
        base_longitude = gongcanMap[key][0]
        base_latitude = gongcanMap[key][1]
        # add
        base_locations.append([base_longitude, base_latitude])

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
    tops = find_top_similarity(np.array(base_locations))
    return datas, data_labels, tops


def find_top_similarity(base_locations):
    mMatrix = base_locations
    # k is 14
    cluster = KMeans(n_clusters=14, random_state=10)
    cluster_labels = cluster.fit_predict(mMatrix)
    k_v = {}
    tops = []
    max = 1
    max_number = 0
    for i in range(mMatrix.shape[0]):
        if k_v.get(cluster_labels[i]) is None:
            k_v[cluster_labels[i]] = 0
        k_v[cluster_labels[i]] += 1
        if k_v[cluster_labels[i]] > max:
            max = k_v[cluster_labels[i]]
            max_number = cluster_labels[i]
    for i in range(mMatrix.shape[0]):
        if cluster_labels[i] == max_number:
            tops.append(i)
    return tops


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


def top_similarity(datas, data_labels, sorted_group_median):
    return datas


def top_similarity_data(tops, datas, data_labels):
    similar_data = datas[tops[0]]
    similar_label = data_labels[tops[0]]
    for index in range(tops[0] + 1, len(datas)):
        # merge
        if index in tops:
            similar_data = np.vstack((datas[index], similar_data))
            similar_label = np.vstack((data_labels[index], similar_label))
    return similar_data, similar_label


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

    datas, data_labels, tops = process_data(dataDf)
    average_errors = []
    group_medians = {}
    for index in range(len(datas)):
        rfc = RandomForestClassifier()
        average_error = train(rfc, datas[index], data_labels[index])
        median = np.median(average_error)
        group_medians[index] = median
        average_errors.append(average_error)

    # top similarity data
    top_similarity_data, top_similarity_label = top_similarity_data(tops, datas, data_labels)

    average_errors = []
    for index in range(len(datas)):
        rfc = RandomForestClassifier(max_depth=2, random_state=0)
        if index in tops:
            average_error = train(rfc, datas[index], data_labels[index],
                                  top_data=top_similarity_data, top_data_label=top_similarity_label)
        else:
            average_error = train(rfc, datas[index], data_labels[index])
        average_errors.append(average_error)
    color = 'r'
    make_picture(average_errors, range(len(datas)), color)