[Distributed Tensorflow](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/distributed_runtime#distributed-tensorflow)
using [gRPC](http://www.grpc.io/)for inter-process communication.

it doesn't support distribute training yet. Compared with other open source libraries, like Caffe, TF's data graph structure is
more suitble for cross-machine tasks.

Accoreding to http://static.googleusercontent.com/media/research.google.com/en//people/jeff/BayLearn2015.pdf，both data and model parallel are
needed to train large and powerful models quickly.

## notes about [Distributed Tensorflow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/how_tos/distributed/index.md)



## Reference
 - [Tensorflow whitepaper](http://download.tensorflow.org/paper/whitepaper2015.pdf)
 
