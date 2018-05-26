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


def get_grid_id(lon, lat):
    x = floor(get_distance(minLatitude, minLongitude, minLatitude, lon) / 20)
    y = floor(get_distance(minLatitude, maxLongitude, lat, maxLongitude) / 20)
    return x + y * colGridSize


def get_gongcan_map(gongcanDf):
    tmpMap = {}
    for i in range(len(gongcanDf)):
        key = gongcanDf.iloc[i]['RNCID'] + '_' + gongcanDf.iloc[i]['CellID']
        value = [gongcanDf.iloc[i]['Longitude'], gongcanDf.iloc[i]['Latitude']]
        tmpMap[key] = value
    return tmpMap


def process_data(dataDf):
    labels = np.zeros((size, 3))
    data = np.zeros((size, 23))
    # Statistic the each point
    for i in range(size):
        lon = dataDf.iloc[i]['Longitude']
        lat = dataDf.iloc[i]['Latitude']
        gridId = get_grid_id(lon, lat)
        labels[i][0] = gridId
        labels[i][1] = lon
        labels[i][2] = lat
        num_connected = dataDf.iloc[i]['Num_connected']
        data[i][0] = dataDf.iloc[i]['IMSI']
        data[i][1] = dataDf.iloc[i]['MRTime']
        for j in range(num_connected):
            location = gongcanMap.get(
                dataDf.iloc[i]['RNCID_' + str(j + 1)] + '_' + dataDf.iloc[i]['CellID_' + str(j + 1)])
            if location is not None:
                data[i][3 * j + 2] = location[0]
                data[i][3 * j + 3] = location[1]
                data[i][3 * j + 4] = dataDf.iloc[i]['RSSI_' + str(j + 1)]
    return data, labels


def get_center_location(grid_id):
    lon = (grid_id % colGridSize + 0.5) / colGridSize * (maxLongitude - minLongitude) + minLongitude
    lat = (floor(grid_id / colGridSize) + 0.5) / rowGridSize * (maxLatitude - minLatitude) + minLatitude
    return [lon, lat]


def evaluate_classifier(y_pred, y_test, max_grids):
    keys = list(max_grids.keys())
    dt = 0
    d = 0
    t = 0
    for i in range(len(keys)):
        dt = 0
        d = 0
        t = max_grids[keys[i]]
        for j in range(len(y_pred)):
            if int(keys[i]) == y_pred[j] == y_test[j, 0]:
                dt += 1
            if int(keys[i]) == y_pred[j]:
                d += 1
    if d == 0:
        precision = -1
    else:
        precision = dt / d
    if t == 0:
        recall = -1
    else:
        recall = dt / t
    if precision + recall == 0:
        f_measurement = -1
    else:
        f_measurement = 2 * precision * recall / (precision + recall)
    return round(precision, 2), round(recall, 2), round(f_measurement, 2)


def init_grip_number(y_test):
    grid_number_map = {}
    # initialize the grid
    for i in range(int(get_grid_id(maxLongitude, maxLatitude))):
        grid_number_map[str(i)] = 0

    for i in range(len(y_test)):
        grid_number_map[str(int(y_test[i][0]))] = grid_number_map[str(int(y_test[i][0]))] + 1
    return grid_number_map


def train(classifier):
    distances = []
    scores = []
    for number in range(10):
        # split data
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=number)
        classifier.fit(X_train[:, 2:], y_train[:, 0])
        # get grid number map
        grid_number_map = init_grip_number(y_test)
        max_grids = get_max_grids(grid_number_map)
        y_pred = classifier.predict(X_test[:, 2:])

        # for p in range(len(X_test)):


        # evaluate the model
        scores.append(evaluate_classifier(y_pred, y_test, max_grids))
        tmp = []
        for index in range(len(y_pred)):
            tmp.append(get_distance(y_test[index][1], y_test[index][2],
                                    get_center_location(y_pred[index])[0],
                                    get_center_location(y_pred[index])[1]))
        distances.append(sorted(tmp))
    errors = []
    matrix = np.array(distances)
    for index in range(matrix.shape[1]):
        errors.append(np.mean(matrix[:, index]))
    return errors, scores


def get_max_grids(grid_number_map):
    grid_number_map = sorted(grid_number_map.items(), key=lambda d: d[1], reverse=True)
    max_grids = {}
    for i in range(10):
        max_grids[grid_number_map[i][0]] = grid_number_map[i][1]
    return max_grids


def make_picture(errors, labels, colors):
    plt.figure(figsize=(15, 8))
    x = range(1220)
    for index in range(len(errors)):
        plt.plot(x, errors[index], label=labels[index], linewidth=0.5, color=colors[index], marker='o',
                 markerfacecolor='blue', markersize=1)
    plt.xlabel('number')
    plt.ylabel('average error')
    plt.title('Average error probability distribution diagram')
    plt.legend()
    plt.show()


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

    # compute the grid row and col number
    colGridSize = floor(get_distance(minLatitude, minLongitude, minLatitude, maxLongitude) / 20)
    rowGridSize = floor(get_distance(minLatitude, minLongitude, maxLatitude, minLongitude) / 20)

    data, labels = process_data(dataDf)

    average_errors = []
    gnb = GaussianNB()
    neigh = KNeighborsClassifier(n_neighbors=3)
    dtc = DecisionTreeClassifier(random_state=0)
    abc = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    rfc = RandomForestClassifier(max_depth=2, random_state=0)
    bc = BaggingClassifier()
    gbc = GradientBoostingClassifier()
    # classifiers = [gnb, neigh, dtc, abc, rfc, bc, gbc]
    classifiers = [neigh]
    for classifier in classifiers:
        start = time.time()
        errors, scores = train(classifier)
        average_errors.append(errors)
        end = time.time()
        print(scores)
        print(end - start, 's')
    # # draw errors
    # x = range(0, 1220)
    # labels = ['GaussianNB', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'AdaBoostClassifier',
    #           'RandomForestClassifier', 'BaggingClassifier', 'GradientBoostingClassifier']
    # colors = ['r', 'g', 'r', 'c', 'm', 'y', 'k']
    # make_picture(average_errors, labels, colors)