

import pickle
with open('logs/events.out.tfevents.1611284542.FKHAN04-GWY1', 'rb') as file:
    history = pickle.load(file)

print(history)
from matplotlib import pyplot as plt

plt.plot(history['losses'])
# plt.plot(history['val_loss'])
# plt.plot(history['mean_squared_error'])
# plt.plot(history['mean_absolute_error'])
# plt.plot(history['mean_absolute_percentage_error'])
# plt.ylabel('loss')
# plt.xlabel('epoch')
# # plt.legend(['train', 'test'], loc='upper left')
plt.show()