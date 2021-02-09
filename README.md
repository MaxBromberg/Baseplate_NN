## Basic Neural Network Playground
This repository serves to reproduce the fundaments of neural networks with basic Python (No tensorflow, keras, etc), 
inspired in large part (and reproduced entirely from) Grant Sanderson's ([3Blue1Brown][1]) neural network [playlist][2]
, with minimal plagerism from Michael Nielsen's [book][3], as glimpsed in the linked playlist. 

Neural_Ney.py hosts the NeuralNet class, a brief example of the presently petit program below:

```
import Neural_Net as NN
import numpy as np

layer_dims = [100, 20, 8, 2]  # Automatically densely connected 
test_net = NN.NeuralNet(layer_dims, "logistic")
test_net.log_network_shapes()  # prints weight, node, bias dimensions

# "Data" creation
virtual_samples = 100
fake_data = np.array([(np.random.rand(test_net.NN_shape[0]), np.random.rand(test_net.NN_shape[-1])) for _ in range(virtual_samples)], dtype=object)
fake_test_data = np.array([(np.random.rand(test_net.NN_shape[0]), np.random.rand(test_net.NN_shape[-1])) for _ in range(virtual_samples)], dtype=object)

test_net.SGD(fake_data, epochs=10, mini_batch_size=int(0.1*fake_data.shape[0]), test_data=fake_test_data)
```
<!--- Links -->
[1]: https://www.youtube.com/c/3blue1brown/featured
[2]: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
[3]: http://neuralnetworksanddeeplearning.com/chap1.html
