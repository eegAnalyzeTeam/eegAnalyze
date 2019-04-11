import pandas as pd


# 从特征名字获取通道名字
def get_channel(features):
    temp_res = []
    for x in features:
        temp = str(x).split('_')[0]
        temp_res.append(temp)
    return list(set(temp_res))


# 读取选择的特征中涉及的特征名字及通道名字
def get_colums_tree(name):
    feature_3s = pd.read_csv(name).columns.values.tolist()

    print(len(feature_3s[1:]))

    channel = get_channel(feature_3s[1:])
    print(channel)
    print(len(channel))

    df = pd.DataFrame(feature_3s[1:])
    df.to_csv('analyze_features.csv', index=False, header=False)


def start():
    get_colums_tree('test_sklearn_ExtraTreesClassifier_4.csv')
