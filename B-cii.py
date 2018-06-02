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


def get_normalize_mean(arr):
    mean = np.mean(arr)
    if abs(mean - 0) > 0.000001:
        normalize = (arr[-1] - mean) / mean
    else:
        normalize = 0
    return [normalize]


def fun(p, x):
    k, b = p
    return k * x + b


def err(p, x, y):
    return fun(p, x) - y


def get_trend(p0, x1, y1):
    return leastsq(err, p0, args=(x1, y1))


def get_data_by_time(vip_df, data_time):
    df = vip_df.set_index('sldat')
    return df[data_time].reset_index()


def get_buy_count_by_time(vip_df, data_time):
    data = vip_df.set_index('sldat')
    df = data[data_time].reset_index()
    return len(df.groupby('sldat').indices.keys())


def get_all_time_data(vipDf, data_time):
    df = vipDf.set_index('sldat')
    return df[data_time[0]: data_time[2]].reset_index()


def get_count_by_time(df, data_time):
    values = []
    for i in range(len(data_time) - 1):
        try:
            values.append(get_buy_count_by_time(df, data_time[i]))
        except:
            values.append(0)
    return values


def get_amt_by_time(df, data_time):
    values = []
    for i in range(len(data_time) - 1):
        try:
            values.append(get_data_by_time(df, data_time[i])['amt'].sum())
        except:
            values.append(0)
    return values


def get_day_by_time(df, data_time):
    values = []
    for i in range(len(data_time) - 1):
        try:
            df = get_data_by_time(df, data_time[i]).set_index('sldat').resample('D').mean()
            values.append(len(df[df['uid'] > 0]))
        except:
            values.append(0)
    return values


def get_all_more_than_two_buy_by_time(df, col, data_time):
    try:
        vip_dict = {}
        data = get_all_time_data(df, data_time)[col]
        count = 0
        for index in range(len(data)):
            try:
                count = count + vip_dict[data.iloc[index]]
            except:
                vip_dict[data.iloc[index]] = 1
        return count
    except:
        return 0


def get_all_more_than_two_day_by_time(df, col, data_time):
    try:
        vip_dict = {}
        data = get_all_time_data(df, data_time).set_index('sldat')
        vip_df = data[col]
        repeat_count = 0
        count = 0
        for index in range(len(vip_df)):
            try:
                vip_dict[vip_df.iloc[index]] += 1
            except:
                vip_dict[vip_df.iloc[index]] = 1
        for key in vip_dict:
            if vip_dict[key] > 1:
                repeat_count += vip_dict[key]
            count += 1
        return repeat_count, count
    except:
        return 0


def get_column_by_time(df, col, data_time):
    values = []
    for i in range(len(data_time) - 1):
        try:
            values.append(get_data_by_time(df, data_time[i])[col].drop_duplicates().count())
        except:
            values.append(0)
    return values


def get_last_month_record(statistic, data_months):
    last_month_ui = {}
    items = tradeDf.set_index('sldat')[data_months[3]]
    for index, row in items.iterrows():
        key = tuple([row[statistic[0]], row[statistic[1]]])
        last_month_ui[key] = True
    return last_month_ui


def get_type_1_1_feature(months):
    statistics = ['vipno', 'bndno', 'dptno', 'pluno', ['vipno', 'bndno'],
                  ['vipno', 'dptno'], ['vipno', 'pluno'], ['bndno', 'dptno']]
    feature_type_1_1 = {}
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        for index in range(1, len(itemNos)):
            itemDf = items.get_group(itemNos[index])
            buy_count = get_count_by_time(itemDf, months)
            amt_count = get_amt_by_time(itemDf, months)
            day_count = get_day_by_time(itemDf, months)
            feature = buy_count + amt_count + day_count + [sum(buy_count), sum(amt_count), sum(day_count)]
            feature_type_1_1[itemNos[index]] = feature
    return feature_type_1_1


def get_type_1_21_feature(months):
    statistics = ['vipno']
    columns = ['pluno', 'bndno', 'dptno']
    features_type_1_21 = {}
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        for index in range(0, len(itemNos)):
            itemDf = items.get_group(itemNos[index])
            i_count = get_column_by_time(itemDf, columns[0], months)
            b_count = get_column_by_time(itemDf, columns[1], months)
            c_count = get_column_by_time(itemDf, columns[2], months)
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
            i_count = get_column_by_time(itemDf, columns[0], months)
            feature = i_count + [sum(i_count)]
            key = itemNos[index]
            features_type_1_22[key] = feature
    return features_type_1_22


def get_type_1_3_feature(months):
    statistics = ['bndno', 'dptno', 'pluno']
    columns = ['vipno']
    features_type_1_3 = {}
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        for index in range(0, len(itemNos)):
            itemDf = items.get_group(itemNos[index])
            i_count = get_column_by_time(itemDf, columns[0], months)
            feature = i_count + [sum(i_count)]
            key = itemNos[index]
            features_type_1_3[key] = feature
    return features_type_1_3


def get_type_2_11_feature(months):
    statistics = ['vipno', 'bndno', 'dptno', 'pluno', ['vipno', 'bndno'],
                  ['vipno', 'dptno'], ['vipno', 'pluno'], ['bndno', 'dptno']]
    feature_type_2_1 = {}
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        for index in range(0, len(itemNos)):
            feature = []
            itemDf = items.get_group(itemNos[index])
            buy_count = get_count_by_time(itemDf, months)
            amt_count = get_amt_by_time(itemDf, months)
            day_count = get_day_by_time(itemDf, months)
            buy_feature = [np.mean(buy_count), np.std(buy_count), np.max(buy_count), np.median(buy_count)]
            amt_feature = [np.mean(amt_count), np.std(amt_count), np.max(amt_count), np.median(amt_count)]
            day_feature = [np.mean(day_count), np.std(day_count), np.max(day_count), np.median(day_count)]
            feature = feature + buy_count + amt_count + day_count + buy_feature + amt_feature + day_feature
            feature_type_2_1[itemNos[index]] = feature
    return feature_type_2_1


def get_type_2_12_feature(months):
    statistics = ['vipno']
    columns = ['pluno', 'bndno', 'dptno']
    features_type_2_12 = {}
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        for index in range(0, len(itemNos)):
            feature = []
            itemDf = items.get_group(itemNos[index])
            i_count = get_column_by_time(itemDf, columns[0], months)
            b_count = get_column_by_time(itemDf, columns[1], months)
            c_count = get_column_by_time(itemDf, columns[2], months)
            buy_feature = [np.mean(i_count), np.std(i_count), np.max(i_count), np.median(i_count)]
            amt_feature = [np.mean(b_count), np.std(b_count), np.max(b_count), np.median(b_count)]
            day_feature = [np.mean(c_count), np.std(c_count), np.max(c_count), np.median(c_count)]
            feature = feature + i_count + b_count + c_count + buy_feature + amt_feature + day_feature
            key = itemNos[index]
            features_type_2_12[key] = feature
    return features_type_2_12


def get_type_2_13_feature(months):
    statistics = ['bndno', 'dptno']
    columns = ['pluno']
    features_type_2_13 = {}
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        for index in range(0, len(itemNos)):
            itemDf = items.get_group(itemNos[index])
            i_count = get_column_by_time(itemDf, columns[0], months)
            buy_feature = [np.mean(i_count), np.std(i_count), np.max(i_count), np.median(i_count)]
            feature = i_count + buy_feature
            key = itemNos[index]
            features_type_2_13[key] = feature
    return features_type_2_13


def get_type_2_2_feature(months):
    statistics = ['pluno', 'bndno', 'dptno']
    features_type_2_2 = {}
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        for index in range(0, len(itemNos)):
            itemDf = items.get_group(itemNos[index])
            vips = itemDf.groupby('vipno')
            vipNos = list(vips.indices.keys())
            buy_counts = amt_counts = day_counts = []
            for j in range(0, len(vipNos)):
                # single feature
                vipDf = vips.get_group(vipNos[j])
                buy_counts.append(sum(get_count_by_time(vipDf, months)))
                amt_counts.append(sum(get_amt_by_time(vipDf, months)))
                day_counts.append(sum(get_day_by_time(vipDf, months)))
            buy_feature = [np.mean(buy_counts), np.std(buy_counts), np.max(buy_counts), np.median(buy_counts)]
            amt_feature = [np.mean(amt_counts), np.std(amt_counts), np.max(amt_counts), np.median(amt_counts)]
            day_feature = [np.mean(day_counts), np.std(day_counts), np.max(day_counts), np.median(day_counts)]
            feature = buy_feature + amt_feature + day_feature
            key = itemNos[index]
            features_type_2_2[key] = feature
    return features_type_2_2


def get_type_2_3_feature(months):
    statistics = ['vipno']
    features_type_2_3 = {}
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        for index in range(0, len(itemNos)):
            itemDf = items.get_group(itemNos[index])
            groups = ['pluno', 'bndno', 'dptno']
            buys = amts = days = []
            for j in range(len(groups)):
                pros = itemDf.groupby(groups[j])
                prosNos = list(pros.indices.keys())
                buy_counts = amt_counts = day_counts = []
                for k in range(0, len(prosNos)):
                    # single feature
                    proDf = pros.get_group(prosNos[k])
                    buy_counts.append(sum(get_count_by_time(proDf, months)))
                    amt_counts.append(sum(get_amt_by_time(proDf, months)))
                    day_counts.append(sum(get_day_by_time(proDf, months)))
                buys.append([np.mean(buy_counts), np.std(buy_counts), np.max(buy_counts), np.median(buy_counts)])
                amts.append([np.mean(amt_counts), np.std(amt_counts), np.max(amt_counts), np.median(amt_counts)])
                days.append([np.mean(day_counts), np.std(day_counts), np.max(day_counts), np.median(day_counts)])
            buy_feature = buys[0] + buys[1] + buys[2]
            amt_feature = amts[0] + amts[1] + amts[2]
            day_feature = days[0] + days[1] + days[2]
            feature = buy_feature + amt_feature + day_feature
            key = itemNos[index]
            features_type_2_3[key] = feature
    return features_type_2_3


def get_type_3_11_feature(days):
    statistics = ['vipno', 'bndno', 'dptno', 'pluno', ['vipno', 'bndno'],
                  ['vipno', 'dptno'], ['vipno', 'pluno'], ['bndno', 'dptno']]
    feature_type_3_11 = {}
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        for index in range(1, len(itemNos)):
            itemDf = items.get_group(itemNos[index])
            buy_count = get_count_by_time(itemDf, days)
            amt_count = get_amt_by_time(itemDf, days)
            day_count = get_day_by_time(itemDf, days)
            feature = [sum(buy_count), sum(amt_count), sum(day_count)]
            feature_type_3_11[itemNos[index]] = feature
    return feature_type_3_11


def get_type_4_11_feature(months):
    statistics = ['vipno', 'bndno', 'dptno', 'pluno', ['vipno', 'bndno'],
                  ['vipno', 'dptno'], ['vipno', 'pluno'], ['bndno', 'dptno']]
    feature_type_4_11 = {}
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        for index in range(1, len(itemNos)):
            itemDf = items.get_group(itemNos[index])
            buy_count = get_count_by_time(itemDf, months)
            amt_count = get_amt_by_time(itemDf, months)
            day_count = get_day_by_time(itemDf, months)
            x1 = np.array([i for i in range(1, len(buy_count) + 1)]).astype(float)
            buy_trend = get_trend([1, 1], x1, np.array(buy_count).astype(float))
            amt_trend = get_trend([1, 1], x1, np.array(amt_count).astype(float))
            day_trend = get_trend([1, 1], x1, np.array(day_count).astype(float))
            feature = [buy_trend[0][0], amt_trend[0][0], day_trend[0][0]]
            feature_type_4_11[itemNos[index]] = feature
    return feature_type_4_11


def get_type_4_121_feature(months):
    statistics = ['vipno']
    columns = ['pluno', 'bndno', 'dptno']
    feature_type_4_121 = {}
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        for index in range(0, len(itemNos)):
            itemDf = items.get_group(itemNos[index])
            i_count = get_column_by_time(itemDf, columns[0], months)
            b_count = get_column_by_time(itemDf, columns[1], months)
            c_count = get_column_by_time(itemDf, columns[2], months)
            x1 = np.array([i for i in range(1, len(i_count) + 1)]).astype(float)
            i_trend = get_trend([1, 1], x1, np.array(i_count).astype(float))
            b_trend = get_trend([1, 1], x1, np.array(b_count).astype(float))
            c_trend = get_trend([1, 1], x1, np.array(c_count).astype(float))
            feature = [i_trend[0][0], b_trend[0][0], c_trend[0][0]]
            feature_type_4_121[itemNos[index]] = feature
    return feature_type_4_121


def get_type_4_122_feature(months):
    statistics = ['bndno', 'dptno']
    columns = ['pluno']
    features_type_4_122 = {}
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        for index in range(0, len(itemNos)):
            itemDf = items.get_group(itemNos[index])
            i_count = get_column_by_time(itemDf, columns[0], months)
            x1 = np.array([i for i in range(1, len(i_count) + 1)]).astype(float)
            i_trend = get_trend([1, 1], x1, np.array(i_count).astype(float))
            features_type_4_122[itemNos[index]] = [i_trend[0][0]]
    return features_type_4_122


def get_type_4_13_feature(months):
    statistics = ['bndno', 'dptno', 'pluno']
    columns = ['vipno']
    features_type_4_13 = {}
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        for index in range(0, len(itemNos)):
            itemDf = items.get_group(itemNos[index])
            i_count = get_column_by_time(itemDf, columns[0], months)
            x1 = np.array([i for i in range(1, len(i_count) + 1)]).astype(float)
            i_trend = get_trend([1, 1], x1, np.array(i_count).astype(float))
            features_type_4_13[itemNos[index]] = [i_trend[0][0]]
    return features_type_4_13


def get_type_4_221_feature(months):
    statistics = ['vipno']
    columns = ['pluno', 'bndno', 'dptno']
    feature_type_4_221 = {}
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        for index in range(0, len(itemNos)):
            itemDf = items.get_group(itemNos[index])
            i_count = get_column_by_time(itemDf, columns[0], months)
            b_count = get_column_by_time(itemDf, columns[1], months)
            c_count = get_column_by_time(itemDf, columns[2], months)
            feature = get_normalize_mean(i_count) + get_normalize_mean(b_count) + get_normalize_mean(c_count)
            feature_type_4_221[itemNos[index]] = feature
    return feature_type_4_221


def get_type_4_222_feature(months):
    statistics = ['bndno', 'dptno']
    columns = ['pluno']
    features_type_4_222 = {}
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        for index in range(0, len(itemNos)):
            itemDf = items.get_group(itemNos[index])
            i_count = get_column_by_time(itemDf, columns[0], months)
            features_type_4_222[itemNos[index]] = get_normalize_mean(i_count)
    return features_type_4_222


def get_type_4_23_feature(months):
    statistics = ['bndno', 'dptno', 'pluno']
    columns = ['vipno']
    features_type_4_23 = {}
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        for index in range(0, len(itemNos)):
            itemDf = items.get_group(itemNos[index])
            i_count = get_column_by_time(itemDf, columns[0], months)
            features_type_4_23[itemNos[index]] = get_normalize_mean(i_count)
    return features_type_4_23


def get_type_4_31_feature(months):
    statistics = ['bndno', 'dptno', 'pluno']
    columns = ['vipno']
    features_type_4_31 = {}
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        for index in range(0, len(itemNos)):
            itemDf = items.get_group(itemNos[index])
            i_count = get_all_more_than_two_buy_by_time(itemDf, columns[0], months)
            day_count, count = get_all_more_than_two_day_by_time(itemDf, columns[0], months)
            buy_count = get_count_by_time(itemDf, months)
            if sum(buy_count) == 0:
                buy_ratio_feature = 0.0
            else:
                buy_ratio_feature = i_count / sum(buy_count)
            if count == 0:
                day_ratio_feature = 0.0
            else:
                day_ratio_feature = day_count / count
            features_type_4_31[itemNos[index]] = [float(i_count)] + [float(buy_ratio_feature)] + \
                                                 [float(day_count)] + [day_ratio_feature]
    return features_type_4_31


def get_train_data(data_last_month_ui):
    features = []
    labels = []
    infos = []
    # get group by ui data
    type_1_1 = type_1_1_feature
    vip = type_1_21_feature
    type_2_11 = type_2_11_feature
    type_2_3 = type_2_3_feature
    type_4_221 = type_4_221_feature
    for key in uiKeys:
        try:
            ui_feature = type_1_1[key] + type_2_11[key]
        except KeyError:
            ui_feature = [0] * 12 + [0] * 21
        try:
            u_feature = type_1_1[key[0]] + type_2_11[key[0]] + type_2_3[key[0]] + type_4_221[key[0]]
        except KeyError:
            u_feature = [0] * 12 + [0] * 21 + [0] * 36 + [0] * 3
        try:
            i_feature = type_1_1[key[1]] + type_2_11[key[1]]
        except KeyError:
            i_feature = [0] * 12 + [0] * 21
        feature = ui_feature + u_feature + i_feature + vip[key[0]]
        features.append(feature)
        infos.append(key)
        try:
            labels.append(data_last_month_ui[key])
        except KeyError:
            labels.append(False)
    return infos, np.array(features), np.array(labels)


def get_test_data(data_last_month_ui):
    features = []
    labels = []
    infos = []
    # get group by ui data
    type_1_1 = test_type_1_1_feature
    type_2_11 = test_type_2_11_feature
    type_1_21 = test_type_1_21_feature
    type_2_3 = test_type_2_3_feature
    type_4_221 = test_type_4_221_feature
    for key in uiKeys:
        try:
            ui_feature = type_1_1[key] + type_2_11[key]
        except KeyError:
            ui_feature = [0] * 12 + [0] * 21
        try:
            u_feature = type_1_1[key[0]] + type_2_11[key[0]] + type_2_3[key[0]] + type_4_221[key[0]]
        except KeyError:
            u_feature = [0] * 12 + [0] * 21 + [0] * 36 + [0] * 3
        try:
            i_feature = type_1_1[key[1]] + type_2_11[key[1]]
        except KeyError:
            i_feature = [0] * 12 + [0] * 21
        feature = ui_feature + u_feature + i_feature + type_1_21[key[0]]
        features.append(feature)
        infos.append(key)
        try:
            labels.append(data_last_month_ui[key])
        except KeyError:
            labels.append(False)
    return infos, np.array(features), np.array(labels)


def write_predict(features, infos, y_pred, my_number, work_number, classifier_name):
    for index in range(len(features)):
        with open('output/' + my_number + '_' + work_number + '_' + classifier_name + '.txt', 'a+') as f:
            content = ''
            for nos in infos[index]:
                content = content + str(nos) + ','
            content = content + str(y_pred[index]) + '\n'
            f.write(content)


if __name__ == "__main__":
    my_number = '1552730'
    work_number = '2b'
    tradeDf = pd.read_csv('trade_new_part.csv', header=0)
    # data pre process
    tradeDf['sldat'] = pd.to_datetime(tradeDf['sldat'])
    tradeDf['bndno'] = tradeDf['bndno'].fillna(-1).astype(int)

    uiDf = tradeDf.groupby(['vipno', 'bndno'])
    itemNos = list(uiDf.indices.keys())
    uiKeys = []
    for index in range(0, len(itemNos)):
        uiKeys.append(itemNos[index])
    # step 1
    months = ['2016-2', '2016-3', '2016-4', '2016-5']
    last_month_ui = get_last_month_record(['vipno', 'bndno'], months)
    type_1_1_feature = get_type_1_1_feature(months)
    type_1_21_feature = get_type_1_21_feature(months)
    type_2_11_feature = get_type_2_11_feature(months)
    type_2_3_feature = get_type_2_3_feature(months)
    type_4_221_feature = get_type_4_221_feature(months)
    info, features, labels = get_train_data(last_month_ui)
    #
    test_months = ['2016-4', '2016-5', '2016-6', '2016-7']
    test_last_month_ui = get_last_month_record(['vipno', 'bndno'], test_months)
    test_type_1_1_feature = get_type_1_1_feature(test_months)
    test_type_1_21_feature = get_type_1_21_feature(test_months)
    test_type_2_11_feature = get_type_2_11_feature(test_months)
    test_type_2_3_feature = get_type_2_3_feature(test_months)
    test_type_4_221_feature = get_type_4_221_feature(test_months)
    test_info, test_features, test_labels = get_test_data(test_last_month_ui)

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
        y_pred = classifier.predict(test_features)
        end = time.time()
        print(end - start, 's', classifier_name + ' precision: ', round((y_pred == test_labels).sum() / len(y_pred), 2))
        # write
        # write_predict(test_features, test_info, y_pred, my_number, work_number, classifier_name)
