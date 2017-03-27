from numpy import *
import numpy as np

def sigmoid(x):
    return 1.0/(1.0+exp(-x))

def load_data():
    train_data = zeros((32561,124))
    train_label = ones((32561,1))
    test_data = zeros((16281,124))
    test_label = ones((16281,1))

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

def loss(X , Y , w):
    tmp1 = float(np.dot(np.dot(w.transpose(), X),Y))
    tmp2 = float(np.sum(np.dot(w.transpose() , X)))
    return tmp1-tmp2


def IRLSmethod(dataMatIn , classLabels):
    dataMatrix = mat(dataMatIn)
    #print dataMatrix
    labelMat = mat(classLabels)
    n , m = shape(dataMatrix)
    R = zeros((n,n))
    Rnn = mat(R)
    eps = 0.0001
    weights_old = mat(0.01*ones((m,1)))
    iternum = 0
    while True:
		iternum += 1
		Y = sigmoid(dataMatrix*weights_old)
		#print shape(Y)
		for i in range(n):
			Rnn[i,i] = Y[i,0]*(1-Y[i,0])

		#print 'rnn' , shape(Rnn) , Rnn
		temp1 = dataMatrix.transpose()*Rnn*dataMatrix
		#print 'temp1' , shape(temp1) , temp1
		temp2 = dataMatrix.transpose()*Rnn
		#print 'temp2' , shape(temp2) , temp2
		RnnInverse = mat(zeros((n , n)))
		for iv in range(n):
			RnnInverse[iv ,iv] = 1.0/Rnn[iv,iv]
		temp3 = dataMatrix*weights_old-RnnInverse*(Y-labelMat)
		temp1 = temp1 + mat(np.eye(124) * np.min(temp1) * 0.0001)
		weights_new = temp1.I * temp2 * temp3
		suberror = weights_new - weights_old
		print np.max(suberror) , loss(dataMatrix.T , labelMat , weights_new)
		weights_old = weights_new

def main():
    X , Y , X_ , Y_ = load_data()
    IRLSmethod(X , Y)

if __name__ == "__main__":
    main()