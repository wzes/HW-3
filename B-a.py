import pandas as pd
import numpy as np


def get_month_data(vip_df, month):
    df = vip_df.set_index('sldat')
    return df[month].reset_index()


def get_all_month_data(vipDf):
    df = vipDf.set_index('sldat')
    return df[months[0]: months[2]].reset_index()


def get_count_by_month(df):
    values = []
    for i in range(3):
        try:
            values.append(len(get_month_data(df, months[i])))
        except:
            values.append(0)
    return values


def get_amt_by_month(df):
    values = []
    for i in range(3):
        try:
            values.append(get_month_data(df, months[i])['amt'].sum())
        except:
            values.append(0)
    return values


def get_day_by_month(df):
    values = []
    for i in range(3):
        try:
            df1 = get_month_data(df, months[i]).set_index('sldat')
            values.append(len(df1.resample('D').mean()))
        except:
            values.append(0)
    return values


def get_column_by_month(df, col):
    values = []
    for i in range(3):
        try:
            values.append(get_month_data(df, months[i])[col].drop_duplicates().count())
        except:
            values.append(0)
    return values


if __name__ == "__main__":
    tradeDf = pd.read_csv('trade_new.csv', header=0)
    months = ['2016-2', '2016-3', '2016-4', '2016-5']
    # data pre process
    tradeDf['sldat'] = pd.to_datetime(tradeDf['sldat'])
    tradeDf['bndno'] = tradeDf['bndno'].fillna(-1).astype(int)

    # step 1
    statistics = ['vipno', 'bndno', 'dptno', 'pluno', ['vipno', 'bndno'],
                  ['vipno', 'dptno'], ['vipno', 'pluno'], ['bndno', 'dptno']]
    # for item in statistics:
    #     items = tradeDf.groupby(item)
    #     itemNos = list(items.indices.keys())
    #     for index in range(1, len(itemNos)):
    #         itemDf = items.get_group(itemNos[index])
    #         buyCount = get_count_by_month(itemDf)
    #         amtCount = get_amt_by_month(itemDf)
    #         dayCount = get_day_by_month(itemDf)
    #         print(buyCount)
    #         print(amtCount)
    #         print(dayCount)
    #         print(sum(buyCount))
    #         print(sum(amtCount))
    #         print(sum(dayCount))
    #         break

    # step 2
    statistics = ['vipno']
    columns = ['pluno', 'bndno', 'dptno']
    # for item in statistics:
    #     items = tradeDf.groupby(item)
    #     itemNos = list(items.indices.keys())
    #     for index in range(0, len(itemNos)):
    #         for col in columns:
    #             alls = get_column_by_month(items.get_group(itemNos[index]), col)
    #             print(alls)
    #             print(sum(alls))
    #         break

    statistics = ['bndno', 'dptno']
    columns = ['pluno']
    # for item in statistics:
    #     items = tradeDf.groupby(item)
    #     itemNos = list(items.indices.keys())
    #     for index in range(1, len(itemNos)):
    #         all = get_column_by_month(items.get_group(itemNos[index]), columns[0])
    #         print(all)
    #         print(sum(all))
    #         break

    # step 3
    statistics = ['bndno', 'dptno', 'pluno']
    columns = ['vipno']
    # for item in statistics:
    #     items = tradeDf.groupby(item)
    #     itemNos = list(items.indices.keys())
    #     for index in range(1, len(itemNos)):
    #         all = get_column_by_month(items.get_group(itemNos[index]), columns[0])
    #         print(all)
    #         print(sum(all))
    #         break

    # step 4
    statistics = ['vipno', 'bndno', 'dptno', 'pluno', ['vipno', 'bndno'],
                  ['vipno', 'dptno'], ['vipno', 'pluno'], ['bndno', 'dptno']]
    # for item in statistics:
    #     items = tradeDf.groupby(item)
    #     itemNos = list(items.indices.keys())
    #     buyCounts = np.array([[0] * 3] * len(itemNos))
    #     amtCounts = np.array([[0] * 3] * len(itemNos))
    #     dayCounts = np.array([[0] * 3] * len(itemNos))
    #     for index in range(0, len(itemNos)):
    #         itemDf = items.get_group(itemNos[index])
    #         buyCount = get_count_by_month(itemDf)
    #         amtCount = get_amt_by_month(itemDf)
    #         dayCount = get_day_by_month(itemDf)
    #         buyCounts[index] = buyCount
    #         amtCounts[index] = amtCount
    #         dayCounts[index] = dayCount
    #     # mean
    #     for index in range(0, 3):
    #         print(np.mean(buyCounts[:, index]))
    #         print(np.mean(amtCounts[:, index]))
    #         print(np.mean(dayCounts[:, index]))
    #     # std
    #     for index in range(0, 3):
    #         print(np.std(buyCounts[:, index]))
    #         print(np.std(amtCounts[:, index]))
    #         print(np.std(dayCounts[:, index]))
    #     # max
    #     for index in range(0, 3):
    #         print(np.max(buyCounts[:, index]))
    #         print(np.max(amtCounts[:, index]))
    #         print(np.max(dayCounts[:, index]))
    #     # median
    #     for index in range(0, 3):
    #         print(np.median(buyCounts[:, index]))
    #         print(np.median(amtCounts[:, index]))
    #         print(np.median(dayCounts[:, index]))


    # step 5
    statistics = ['pluno', 'bndno', 'dptno']
    # for item in statistics:
    #     items = tradeDf.groupby(item)
    #     itemNos = list(items.indices.keys())
    #     buymeans = np.array([0] * len(itemNos))
    #     buystds = np.array([0] * len(itemNos))
    #     buymaxs = np.array([0] * len(itemNos))
    #     buymedians = np.array([0] * len(itemNos))
    #     amtmeans = np.array([0] * len(itemNos))
    #     amtstds = np.array([0] * len(itemNos))
    #     amtmaxs = np.array([0] * len(itemNos))
    #     amtmedians = np.array([0] * len(itemNos))
    #     daymeans = np.array([0] * len(itemNos))
    #     daystds = np.array([0] * len(itemNos))
    #     daymaxs = np.array([0] * len(itemNos))
    #     daymedians = np.array([0] * len(itemNos))
    #     #
    #     for index in range(0, len(itemNos)):
    #         itemDf = items.get_group(itemNos[index])
    #         vips = itemDf.groupby('vipno')
    #         vipNos = list(vips.indices.keys())
    #         buyCounts = []
    #         amtCounts = []
    #         dayCounts = []
    #         for j in range(0, len(vipNos)):
    #             # single feature
    #             vipDf = vips.get_group(vipNos[j])
    #             buyCount = sum(get_count_by_month(vipDf))
    #             amtCount = sum(get_amt_by_month(vipDf))
    #             dayCount = sum(get_day_by_month(vipDf))
    #             buyCounts.append(buyCount)
    #             amtCounts.append(amtCount)
    #             dayCounts.append(dayCount)
    #         # agg
    #         buymeans[index] = np.mean(np.array(buyCounts))
    #         buystds[index] = np.mean(np.array(buyCounts))
    #         buymaxs[index] = np.max(np.array(buyCounts))
    #         buymedians[index] = np.median(np.array(buyCounts))
    #
    #         amtmeans[index] = np.mean(np.array(amtCounts))
    #         amtstds[index] = np.mean(np.array(amtCounts))
    #         amtmaxs[index] = np.max(np.array(amtCounts))
    #         amtmedians[index] = np.median(np.array(amtCounts))
    #
    #         daymeans[index] = np.mean(np.array(dayCounts))
    #         daystds[index] = np.mean(np.array(dayCounts))
    #         daymaxs[index] = np.max(np.array(dayCounts))
    #         daymedians[index] = np.median(np.array(dayCounts))
    #     break

    # step 6
    statistics = ['vipno']
    for item in statistics:
        items = tradeDf.groupby(item)
        itemNos = list(items.indices.keys())
        buymeans = np.array([[0] * len(itemNos)] * 3)
        buystds = np.array([[0] * len(itemNos)] * 3)
        buymaxs = np.array([[0] * len(itemNos)] * 3)
        buymedians = np.array([[0] * len(itemNos)] * 3)
        amtmeans = np.array([[0] * len(itemNos)] * 3)
        amtstds = np.array([[0] * len(itemNos)] * 3)
        amtmaxs = np.array([[0] * len(itemNos)] * 3)
        amtmedians = np.array([[0] * len(itemNos)] * 3)
        daymeans = np.array([[0] * len(itemNos)] * 3)
        daystds = np.array([[0] * len(itemNos)] * 3)
        daymaxs = np.array([[0] * len(itemNos)] * 3)
        daymedians = np.array([[0] * len(itemNos)] * 3)
        for index in range(0, len(itemNos)):
            itemDf = items.get_group(itemNos[index])
            groups = ['pluno', 'bndno', 'dptno']
            for j in range(len(groups)):
                pros = itemDf.groupby(groups[j])
                prosNos = list(pros.indices.keys())
                buyCounts = []
                amtCounts = []
                dayCounts = []
                for k in range(0, len(prosNos)):
                    # single feature
                    proDf = pros.get_group(prosNos[j])
                    buyCount = sum(get_count_by_month(proDf))
                    amtCount = sum(get_amt_by_month(proDf))
                    dayCount = sum(get_day_by_month(proDf))
                    buyCounts.append(buyCount)
                    amtCounts.append(amtCount)
                    dayCounts.append(dayCount)
                # agg
                buymeans[index][j] = np.mean(np.array(buyCounts))
                buystds[index][j] = np.mean(np.array(buyCounts))
                buymaxs[index][j] = np.max(np.array(buyCounts))
                buymedians[index][j] = np.median(np.array(buyCounts))

                amtmeans[index][j] = np.mean(np.array(amtCounts))
                amtstds[index][j] = np.mean(np.array(amtCounts))
                amtmaxs[index][j] = np.max(np.array(amtCounts))
                amtmedians[index][j] = np.median(np.array(amtCounts))

                daymeans[index][j] = np.mean(np.array(dayCounts))
                daystds[index][j] = np.mean(np.array(dayCounts))
                daymaxs[index][j] = np.max(np.array(dayCounts))
                daymedians[index][j] = np.median(np.array(dayCounts))

            print(buymeans)
            break
        break

    # step 6
