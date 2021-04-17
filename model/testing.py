import matplotlib.pyplot as plt
import scipy.io
import numpy as np
data = scipy.io.loadmat('/home/admins/Desktop/CoW-Computing_on_Wheels-/model/sarl_model/reward.mat')
dot = []
x = data['reward']
n = 100
rewards = [ x[i:i+n] for i in range(0, len(x), n) ]
for i in rewards:
    dot.append(np.mean(i))
    #print(np.mean(i))
    #print(len(i))
print(len(dot))
plt.plot(dot, linestyle='', marker='.')
plt.show()
