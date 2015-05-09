import mnist_loader
training_data, valadation_data, test_data = \
	mnist_loader.load_data_wrapper()

import Network
net = Network.Network([784,30,10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
