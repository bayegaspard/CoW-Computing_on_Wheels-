
data = [2,0,1,3]
#print(data)
#data2 = [print(list(data[0:i+1:1])) for i in range(len(data))]


#def replicate(i):
#    return List+str(i) = [i,i,i,i]
data2 = [(data[i:i+1]*len(data)) for i in range(len(data)) ]

delta = []

def func():
    [print(data2[i][j]) for i, j in zip(range(len(data2)), range(len(data2)))]

#delta = [(i - int(func())) for i in data]

for o in range(len(data2)):
    #print(data2[i])
    for k, l in zip(range(len(data2)), range(len(data2))):
        delta.append(data[k]-data2[o][l])


#for i,j in zip(range(4),range(4)):
 #   del delta[i][j]


print(data2)


print(delta)