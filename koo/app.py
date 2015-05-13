import mnist_loader
training_data, valadation_data, test_data = \
	mnist_loader.load_data_wrapper()
import numpy as np
import Network
#net = Network.Network([784,30,10])
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
import network2
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.5,
			lmbda = 5.0,
			evaluation_data = valadation_data,
			monitor_evaluation_accuracy=True,
			monitor_evaluation_cost=True,
			monitor_training_accuracy=True,
			monitor_training_cost=True)
