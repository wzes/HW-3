import pandas as pd
import numpy as np
from scipy.optimize import leastsq


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


if __name__ == "__main__":
    tradeDf = pd.read_csv('trade_new_part.csv', header=0)
    months = ['2016-2', '2016-3', '2016-4', '2016-5']
    # data pre process
    tradeDf['sldat'] = pd.to_datetime(tradeDf['sldat'])
    tradeDf['bndno'] = tradeDf['bndno'].fillna(-1).astype(int)

    # step 1
    type_1_1_feature = get_type_1_1_feature(months)
    print(type_1_1_feature)
    # step 2
    type_1_21_feature = get_type_1_21_feature(months)
    print(type_1_21_feature)
    #
    type_1_22_feature = get_type_1_22_feature(months)
    print(type_1_22_feature)
    # step 3
    type_1_3_feature = get_type_1_3_feature(months)
    print(type_1_3_feature)

    # step 4
    type_2_11_feature = get_type_2_11_feature(months)
    print(type_2_11_feature)
    type_2_12_feature = get_type_2_12_feature(months)
    print(type_2_12_feature)
    type_2_13_feature = get_type_2_13_feature(months)
    print(type_2_13_feature)

    # step 5
    type_2_2_feature = get_type_2_2_feature(months)
    print(type_2_2_feature)

    type_2_3_feature = get_type_2_3_feature(months)
    print(type_2_3_feature)

    last_week = ['2016-4-24', '2016-4-25', '2016-4-26', '2016-4-27', '2016-4-28', '2016-4-29', '2016-4-30', '2016-5']
    last_month = ['2016-4', '2016-5']
    type_3_111_feature = get_type_3_11_feature(last_week)
    type_3_112_feature = get_type_3_11_feature(last_month)
    print(type_3_111_feature)
    print(type_3_112_feature)
    type_4_11_feature = get_type_4_11_feature(months)
    print(type_4_11_feature)

    type_4_121_feature = get_type_4_121_feature(months)
    print(type_4_121_feature)

    type_4_122_feature = get_type_4_122_feature(months)
    print(type_4_122_feature)

    type_4_13_feature = get_type_4_13_feature(months)
    print(type_4_13_feature)
    type_4_221_feature = get_type_4_221_feature(months)
    print(type_4_221_feature)
    type_4_222_feature = get_type_4_222_feature(months)
    print(type_4_222_feature)
    type_4_23_feature = get_type_4_23_feature(months)
    print(type_4_23_feature)

    type_4_31_feature = get_type_4_31_feature(months)
    print(type_4_31_feature)