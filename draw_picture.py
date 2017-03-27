import matplotlib.pyplot as plt



def plot_noreg():
	itertion = []
	train = []
	test = []
	fr = open('noreg.txt' , 'r')
	for line in fr:
		line = line.strip().split()
		itertion.append(int(line[0]))
		train.append(float(line[1]))
		test.append(float(line[2]))

	plt.plot(itertion , train , "+-" , label="train_accuracy")
	plt.plot(itertion , test , "o-" , label="test_accuracy")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="down")
	plt.show()

def plot_reg():
	itertion = []
	train = []
	test = []
	fr = open('reg_50.txt', 'r')
	for line in fr:
		line = line.strip().split()
		itertion.append(int(line[0]))
		train.append(float(line[1]))
		test.append(float(line[2]))


	plt.plot(itertion, train, "+-", label="train_accuracy")
	plt.plot(itertion, test, "o-", label="test_accuracy")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="down")
	plt.show()

def plot_cmp():
	reg = []
	noreg = []
	fr = open('reg_50.txt', 'r')
	for line in fr:
		line = line.strip().split()
		reg.append(float(line[2]))
	fr.close()
	fr = open('noreg.txt','r')
	for line in fr:
		line = line.strip().split()
		noreg.append(float(line[2]))
	fr.close()
	plt.plot(range(0,7), noreg, "+-", label="noreg_accuracy")
	plt.plot(range(0,7), reg, "o-", label="reg_accuracy")
	plt.xlabel("Iteration")
	plt.ylabel("Test Accuracy")
	plt.legend(loc="best")
	plt.show()

def plot_w():
	reg = []
	noreg = []
	fr = open('reg_50.txt', 'r')
	for line in fr:
		line = line.strip().split()
		reg.append(float(line[4]))
	fr.close()
	fr = open('no_reg.txt','r')
	for line in fr:
		line = line.strip().split()
		noreg.append(float(line[4]))
	fr.close()
	plt.plot(range(0,7), noreg, "+-", label="noreg_w")
	plt.plot(range(0,7), reg, "o-", label="reg_w")
	plt.xlabel("Iteration")
	plt.ylabel("|w|")
	plt.legend(loc="best")
	plt.show()

plot_w()