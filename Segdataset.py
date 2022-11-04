import pandas as pd
import numpy as np

def Segdataset(root=r'abalone.xlsx',type='train',random_state=1):
    data = pd.read_excel(root)
    data = np.array(data)
    np.random.seed(random_state)
    np.random.shuffle(data)

    for i in range(len(data)):
        if data[i, 0] == 'M':
            data[i, 0] = 1
        elif data[i, 0] == 'I':
            data[i, 0] = 0
        else:
            data[i, 0] = -1
    # data=data.astype(np.float)
    data = data.astype(np.float)
    features = data[:, :-1]
    labels = data[:, -1]
    features_train = features[0:3500, :]
    features_test = features[3500:, :]
    labels_train = labels[0:3500]
    labels_test = labels[3500:]

    if type=='train':
        return features_train, labels_train  # 两者路径的列表
    if type=='test':
        return features_test, labels_test