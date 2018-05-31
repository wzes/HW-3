import pandas as pd
import numpy as np
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt


def get_month_data(vip_df, month):
    df = vip_df.set_index('sldat')
    return df[month].reset_index()


def get_buy_count_by_month(vip_df, month):
    data = vip_df.set_index('sldat')
    df = data[month].reset_index()
    return len(df.groupby('sldat').indices.keys())


def get_all_month_data(vipDf):
    df = vipDf.set_index('sldat')
    return df[months[0]: months[2]].reset_index()


def get_count_by_month(df, months):
    values = []
    for i in range(len(months) - 1):
        try:
            values.append(get_buy_count_by_month(df, months[i]))
        except:
            values.append(0)
    return values


def get_amt_by_month(df, months):
    values = []
    for i in range(len(months) - 1):
        try:
            values.append(get_month_data(df, months[i])['amt'].sum())
        except:
            values.append(0)
    return values


def get_day_by_month(df, months):
    values = []
    for i in range(len(months) - 1):
        try:
            df1 = get_month_data(df, months[i]).set_index('sldat')
            values.append(len(df1.resample('D').mean()))
        except:
            values.append(0)
    return values


def get_column_by_month(df, col, months):
    values = []
    for i in range(3):
        try:
            values.append(get_month_data(df, months[i])[col].drop_duplicates().count())
        except:
            values.append(0)
    return values


def get_last_month_record(statistic, months):
    last_month_ui = {}
    items = tradeDf.set_index('sldat')[months[3]]
    for index, row in items.iterrows():
        key = tuple([row[statistic[0]], row[statistic[1]]])
        last_month_ui[key] = True
    return last_month_ui


def get_type_1_1_feature(months):
    statistics = ['vipno', 'bndno', 'dptno', 'pluno', ['vipno', 'bndno'],
                  ['vipno', 'dptno'], ['vipno', 'pluno'], ['bndno', 'dptno']]
    features_type_1_1 = []
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        type_1_1 = {}
        for index in range(1, len(itemNos)):
            itemDf = items.get_group(itemNos[index])
            buyCount = get_count_by_month(itemDf, months)
            amtCount = get_amt_by_month(itemDf, months)
            dayCount = get_day_by_month(itemDf, months)
            feature = buyCount + amtCount + dayCount + [sum(buyCount), sum(amtCount), sum(dayCount)]
            type_1_1[itemNos[index]] = feature
        features_type_1_1.append(type_1_1)
    return features_type_1_1


def get_type_1_21_feature(months):
    statistics = ['vipno']
    columns = ['pluno', 'bndno', 'dptno']
    features_type_1_21 = {}
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        for index in range(0, len(itemNos)):
            itemDf = items.get_group(itemNos[index])
            i_count = get_column_by_month(itemDf, columns[0], months)
            b_count = get_column_by_month(itemDf, columns[1], months)
            c_count = get_column_by_month(itemDf, columns[2], months)
            feature = i_count + b_count + c_count + [sum(i_count), sum(b_count), sum(c_count)]
            key = itemNos[index]
            features_type_1_21[key] = feature
    return features_type_1_21


def get_type_1_22_feature(months):
    statistics = ['bndno', 'dptno']
    columns = ['pluno']
    features_type_1_22 = {}
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        for index in range(0, len(itemNos)):
            itemDf = items.get_group(itemNos[index])
            i_count = get_column_by_month(itemDf, columns[0], months)
            feature = i_count + [sum(i_count)]
            key = itemNos[index]
            features_type_1_22[key] = feature
    return features_type_1_22


def get_train_data(type_1_1_feature, index):
    features = []
    labels = []
    infos = []
    items = type_1_1_feature
    # get group by ui data
    ui = items[index]
    for key in ui.keys():
        features.append(ui[key])
        infos.append(key)
        try:
            labels.append(last_month_ui[key])
        except:
            labels.append(False)
    return infos, np.array(features), np.array(labels)


def get_test_data(type_1_1_feature, index):
    features = []
    infos = []
    items = type_1_1_feature
    # get group by ui data
    ui = items[index]
    for key in ui.keys():
        features.append(ui[key])
        infos.append(key)
    return infos, np.array(features)


def write_predict(features, infos, y_pred, my_number, work_number, classifier_name):
    for index in range(len(features)):
        with open('output/' + my_number + '_' + work_number + '_' + classifier_name + '.txt', 'a+') as f:
            f.write(infos[index][0] + ',' + infos[index][1] + ',' + y_pred[index] + '\n')


if __name__ == "__main__":
    my_number = '1552730'
    work_number = '2ci'
    tradeDf = pd.read_csv('trade_new.csv', header=0, dtype={'vipno': np.object, 'pluno': np.object})
    # data pre process
    tradeDf['sldat'] = pd.to_datetime(tradeDf['sldat'])
    tradeDf['bndno'] = tradeDf['bndno'].fillna(-1).astype(int)

    # step 1
    months = ['2016-2', '2016-3', '2016-4', '2016-5']
    last_month_ui = get_last_month_record(['vipno'], months)
    type_1_1_feature = get_type_1_1_feature(months)
    infos, features, labels = get_train_data(type_1_1_feature, 0)
    #
    test_months = ['2016-4', '2016-5', '2016-6', '2016-7']
    test_last_month_ui = get_last_month_record(['vipno'], test_months)
    test_features = get_type_1_1_feature(test_months)
    test_infos, test_features, test_labels = get_train_data(test_features, 0)
    #
    gnb = GaussianNB()
    neigh = KNeighborsClassifier(n_neighbors=3)
    dtc = DecisionTreeClassifier(random_state=0)
    abc = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    rfc = RandomForestClassifier()
    bc = BaggingClassifier()
    gbc = GradientBoostingClassifier()
    classifiers = [gnb, neigh, dtc, abc, rfc, bc]

    # classifiers = [gnb, neigh, dtc, abc, rfc, bc, gbc]
    for classifier in classifiers:
        start = time.time()
        classifier_name = classifier.__class__.__name__
        classifier.fit(features, labels)
        y_pred = gnb.predict(test_features)
        end = time.time()
        print(end - start, 's')
        # write
        write_predict(test_features, infos, y_pred, my_number, work_number, classifier_name)
