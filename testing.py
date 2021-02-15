# #
# # data = [2,0,1,3]
# # #print(data)
# # #data2 = [print(list(data[0:i+1:1])) for i in range(len(data))]
# #
# #
# # #def replicate(i):
# # #    return List+str(i) = [i,i,i,i]
# # data2 = [(data[i:i+1]*len(data)) for i in range(len(data)) ]
# #
# # delta = []
# #
# # def func():
# #     [print(data2[i][j]) for i, j in zip(range(len(data2)), range(len(data2)))]
# #
# # #delta = [(i - int(func())) for i in data]
# #
# # for o in range(len(data2)):
# #     #print(data2[i])
# #     for k, l in zip(range(len(data2)), range(len(data2))):
# #         delta.append(data[k]-data2[o][l])
# #
# #
# # #for i,j in zip(range(4),range(4)):
# #  #   del delta[i][j]
# #
# # import numpy
# #
# # length = 2
# # width = 2
# # a = [range(16)]
# #a = numpy.reshape(a, (length, width))
# #print(a.reshape((4,4)))
# import numpy as np
# a = [range(16)]
#
# c = np.reshape(a,(4,4))
# x = [4,5,63,1,6,4,7,89,22,4,7,4,9,12,3,7]
# b = np.reshape(x,(4,4))
# v = []
# smallest = []
# for t in range(len(c)):
#     for j in range(len(c)):
#         v.append((c[t][j] + b[t][j])*0.5)
# u = np.reshape(v,(4,4))
# for i in range(len(u)):
#     sort_idx = np.argsort(u[:, i])
#     for k in range(1):
#         smallest.append(sort_idx[k+1])
#
# print(smallest)
# #d = [(c[i][j] + b[i][j]) for i,j in zip(range(len(c)), range(len(c)))]
# #print(v)


import numpy as np
ind = np.random.randint(0, 6)
print("1", str(ind))
ind = np.random.randint(0, 6)
print("2" , str(ind))
ind = np.random.randint(0, 6)
print("3", str(ind))
