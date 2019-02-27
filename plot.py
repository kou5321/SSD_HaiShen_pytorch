import matplotlib.pyplot as plt 

x=[]
y=[]

for line in open('train.log'):
    if line[0]=='N':
        continue
    line = line.strip().split(' || ')
    # print(line)#['iter 300000', 'Loss: 362.4020690917969', 'cur_lr: 1.0000000000000004e-12']
    iter,loss = line[0].split(' ')[1],line[1].split(' ')[1]
    x.append(int(iter))
    y.append(float(loss))
plt.plot(x,y)
plt.show()