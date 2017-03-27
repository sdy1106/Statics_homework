import sys , os
import numpy as np
import math

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def load_data():
    train_data = np.zeros((32561,124))
    train_label = np.ones((32561,1))
    test_data = np.zeros((16281,124))
    test_label = np.ones((16281,1))

    fr = open('a9a' , 'r')
    cnt = 0
    for line in fr:
        line = line.strip().split()
        if float(line[0]) == -1:
            train_label[cnt , 0] = 0
        for ele in line[1:]:
            ele = ele.split(":")
            train_data[cnt , int(ele[0])] = 1
        train_data[0] = 1
        cnt += 1
    fr.close()

    fr = open('a9a.t' , 'r')
    cnt = 0
    for line in fr:
        line = line.strip().split()
        if float(line[0]) == -1:
            test_label[cnt , 0] = 0
        for ele in line[1:]:
            ele = ele.split(":")
            test_data[cnt , int(ele[0])] = 1
        test_data[0] = 1
        cnt += 1
    return train_data , train_label , test_data , test_label

def loss_noreg(X, Y, w):
    # tmp1 = float(np.dot(np.dot(w.transpose(), X),Y))
    # tmp2 = float(np.sum(np.dot(w.transpose() , X)))
    tmp1 = w.T*X*Y
    tmp2 = np.sum(np.log(1+np.exp(w.T*X)))
    return float(tmp1)-tmp2

def loss_reg(X , Y , w , lamda):
    loss1 = loss_noreg(X, Y, w)
    reg = lamda/2.0 *np.linalg.norm(w,ord=2)
    print "loss:" , loss1 , "reg:" , reg
    return loss1-reg

def accuracy(X , Y , w):
    d , N = np.shape(X)
    res = sigmoid(X.T * w) - Y
    #print res
    right = 0.0
    for i in range(N):
        if np.abs(res[i,0]) < 0.5:
            right += 1.0
    return right/float(N)

def training_noreg():
    train_data, train_label, test_data, test_label = load_data()
    #w = np.mat(np.ones((d,1)))
    train_data = np.mat(train_data.transpose())
    train_label = np.mat(train_label)
    test_data = np.mat(test_data.transpose())
    test_label = np.mat(test_label)
    d, N = np.shape(train_data)
    #w = np.mat(np.random.uniform(0, 0.01, (d, 1)))
    w = np.mat(0.01 * np.ones((d, 1)))
    Rnn = np.mat(np.zeros((N , N)))
    #print partial_loss(train_data,train_label,w)
    last_loss = 0
    cnt = 0
    fw = open("no_reg.txt" , "w")
    while True:
        loss = loss_noreg(train_data, train_label, w)
        train_accuracy = accuracy(train_data , train_label , w)
        test_accuracy = accuracy(test_data, test_label, w)
        print cnt , ' train:' ,train_accuracy , ' test:' , test_accuracy , ' loss:' , loss, ' |w|:' , np.linalg.norm(w , 2)
        fw.write(str(cnt)+" "+str(train_accuracy)+" "+str(test_accuracy)+" "+str(loss)+ " " +str(np.linalg.norm(w,2)) +"\n")
        if math.fabs(last_loss-loss) < 1:
            last_loss = loss
            break
        last_loss = loss
        y_ = sigmoid((w.T * train_data).transpose())
        for i in range(N):
            Rnn[i,i] = y_[i,0] * (1-y_[i,0])
        tmp1 = train_data * Rnn * train_data.T
        tmp2 = train_data * Rnn
        Rnn_inv = np.mat(np.zeros((N , N)))
        for i in range(N):
            Rnn_inv[i , i] = 1.0/Rnn[i,i]
        tmp3 = train_data.T*w - Rnn_inv*(y_-train_label)
        #tmp1 = tmp1 + np.mat(np.eye(d) * np.min(tmp1) * 0.0001)
        tmp1_inv,_,_,_ = np.linalg.lstsq(tmp1 , np.eye(d))
        w = tmp1_inv * tmp2 * tmp3
        print w
        cnt += 1
    fw.close()
    return last_loss

def training_reg(lamda):
    train_data, train_label, test_data, test_label = load_data()
    train_data = np.mat(train_data.transpose())
    train_label = np.mat(train_label)
    test_data = np.mat(test_data.transpose())
    test_label = np.mat(test_label)
    d, N = np.shape(train_data)
    Rnn = np.mat(np.zeros((N, N)))
    #w = np.mat(np.random.uniform(0, 0.1, (d, 1)))
    w = np.mat(0.01*np.ones((d,1)))
    last_loss = 0
    cnt = 0
    fw = open("reg_%d.txt" % int(lamda) , "w")

    while True:
        loss = loss_reg(train_data, train_label, w , lamda)
        train_accuracy = accuracy(train_data , train_label , w)
        test_accuracy = accuracy(test_data, test_label, w)
        print cnt , ' train:' ,train_accuracy , ' test:' , test_accuracy , ' loss:' , loss  , ' |w|:' , np.linalg.norm(w , 2)
        fw.write(str(cnt)+" "+str(train_accuracy)+" "+str(test_accuracy)+" "+str(loss)+" " + str(np.linalg.norm(w,2)) +"\n")
        if math.fabs(last_loss-loss) < 1:
            last_loss = loss
            break
        last_loss = loss
        y_ = sigmoid((w.T * train_data).transpose())
        for i in range(N):
            Rnn[i, i] = y_[i, 0] * (1 - y_[i, 0])
        tmp1 = train_data * Rnn * train_data.T + np.mat(np.eye(d)*lamda)
        tmp2 = train_data * Rnn
        Rnn_inv = np.mat(np.zeros((N, N)))
        for i in range(N):
            Rnn_inv[i, i] = 1.0 / Rnn[i, i]
        tmp3 = train_data.T * w - Rnn_inv * (y_ - train_label)
        #tmp1 = tmp1 + np.mat(np.eye(d) * np.min(tmp1) * 0.0001)
        w = tmp1.I * tmp2 * tmp3
        print w
        cnt += 1
        if cnt >= 10:
            break
    fw.close()
    return loss

if __name__ == "__main__":
    b = np.mat([[1,0],[1,1]])
    b = b - np.mat(np.eye(2))

    res ,_ ,_,_= np.linalg.lstsq(b , np.eye(2))
    print res
    print b
    # print a*b

    training_noreg()
    lamda = 50
    #training_reg(lamda)
        #lamda += 0.1

