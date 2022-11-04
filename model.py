from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras import optimizers

def bulid_model():
    model = Sequential()

    model.add(Dense(16, input_shape=(8,)))  # 输入层，28*28=784
    model.add(Activation('softmax'))  # 激活函数是tanh
    # model.add(Dropout(0.5))  # 采用50%的dropout

    model.add(Dense(16))  # 隐藏层节点500个
    model.add(Activation('softmax'))
    # model.add(Dropout(0.5))

    # model.add(Dense(16))  # 隐藏层节点500个
    # model.add(Activation('softmax'))
    # model.add(Dropout(0.5))

    model.add(Dense(1))  # 输出结果是10个类别，所以维度是10

    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.01), metrics=['mae'])
    return model