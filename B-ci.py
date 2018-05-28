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


def is_exist(df):
    try:
        data = len(get_month_data(df, months[3]))
    except:
        data = 0
    return data > 0


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
    for i in range(len(months) - 1):
        try:
            values.append(get_month_data(df, months[i])[col].drop_duplicates().count())
        except:
            values.append(0)
    return values


def get_feature(itemDf, months):
    buyCount = get_count_by_month(itemDf, months)
    amtCount = get_amt_by_month(itemDf, months)
    dayCount = get_day_by_month(itemDf, months)
    feature = buyCount + amtCount + dayCount + [sum(buyCount), sum(amtCount), sum(dayCount)]
    return feature


def get_features(tradeDf, statistic, months):
    features = []
    infos = []
    items = tradeDf.groupby(statistic)
    itemNos = list(items.indices.keys())
    for index in range(1, len(itemNos)):
        itemDf = items.get_group(itemNos[index])
        infos.append(itemNos[index])
        feature = get_feature(itemDf, months)
        features.append(feature)
    return infos, np.array(features)


def train(tradeDf, statistic, months):
    features = []
    labels = []
    infos = []
    items = tradeDf.groupby(statistic)
    itemNos = list(items.indices.keys())
    for index in range(1, len(itemNos)):
        itemDf = items.get_group(itemNos[index])
        feature = get_feature(itemDf, months)
        features.append(feature)
        infos.append(itemNos[index])
        labels.append(is_exist(itemDf))
    return infos, np.array(features), np.array(labels)


def write_predict(features, infos, y_pred, my_number, work_number, classifier_name):
    for index in range(len(features)):
        with open('output/' + my_number + '_' + work_number + '_' + classifier_name + '.txt', 'a+') as f:
            f.write(infos[index][0] + ',' + infos[index][1] + ',' + y_pred[index] + '\n')


if __name__ == "__main__":
    my_number = '1552730'
    work_number = '2b'
    tradeDf = pd.read_csv('trade_new.csv', header=0, dtype={'vipno': np.object, 'pluno': np.object})
    # data pre process
    tradeDf['sldat'] = pd.to_datetime(tradeDf['sldat'])
    tradeDf['bndno'] = tradeDf['bndno'].fillna(-1).astype(int)

    # step 1
    # statistics = ['vipno', 'bndno', 'dptno', 'pluno', ['vipno', 'bndno'],
    #               ['vipno', 'dptno'], ['vipno', 'pluno'], ['bndno', 'dptno']]
    statistic = ['vipno', 'pluno']
    months = ['2016-2', '2016-3', '2016-4', '2016-5']
    infos, features, labels = train(tradeDf, statistic, months)
    gnb = GaussianNB()
    neigh = KNeighborsClassifier(n_neighbors=3)
    dtc = DecisionTreeClassifier(random_state=0)
    abc = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    rfc = RandomForestClassifier()
    bc = BaggingClassifier()
    gbc = GradientBoostingClassifier()
    classifiers = [gnb, neigh, dtc, abc, rfc, bc, gbc]
    for classifier in classifiers:
        start = time.time()
        classifier_name = classifier.__class__.__name__
        classifier.fit(features, labels)
        months = ['2016-5', '2016-6', '2016-7', '2016-8']
        test_features = get_features(tradeDf, statistic, months)
        y_pred = gnb.predict(test_features)
        end = time.time()
        print(end - start, 's')
        # write
        write_predict(test_features, infos, y_pred, my_number, work_number, classifier_name)
