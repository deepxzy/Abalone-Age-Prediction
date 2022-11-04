from Segdataset import Segdataset
from model import bulid_model

import matplotlib.pyplot as plt
model=bulid_model()
features_train, labels_train=Segdataset(root=r'abalone.xlsx',type='train')
features_test, labels_test=Segdataset(root=r'abalone.xlsx',type='test')
history=model.fit(
    features_train, labels_train,epochs=30,batch_size=20,validation_data=(features_test, labels_test)
)

result=model.evaluate(features_test, labels_test)
print(result)

history_dict=history.history
train_loss=history_dict['loss']
test_loss=history_dict['val_loss']
train_mae=history_dict['mae']
test_mae=history_dict['val_mae']
epoches=range(1,len(train_loss)+1)

plt.subplot(1,2,1)
plt.plot(epoches,train_loss,'bo',label='train_loss')
plt.plot(epoches,test_loss,'b',label='test_loss')
plt.title('Train and Test loss')
plt.xlabel('epoches')
plt.ylabel('loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epoches,train_mae,'bo',label='train_mae')
plt.plot(epoches,test_mae,'b',label='test_mae')
plt.title('Train and Test MAE')
plt.xlabel('epoches')
plt.ylabel('MAE')
plt.legend()

plt.show()
