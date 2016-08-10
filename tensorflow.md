[Distributed Tensorflow](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/distributed_runtime#distributed-tensorflow)
using [gRPC](http://www.grpc.io/)for inter-process communication.

it doesn't support distribute training yet. Compared with other open source libraries, like Caffe, TF's data graph structure is more suitble for cross-machine tasks.
According to http://static.googleusercontent.com/media/research.google.com/en//people/jeff/BayLearn2015.pdf， both data and model parallel are needed to train large and powerful models quickly.

## notes about Distributed Tensorflow
王益给的参考：
 1. https://github.com/amygdala/tensorflow-workshop/blob/master/workshop_sections/distributed_tensorflow/README.md
 2. https://storage.googleapis.com/amy-jo/talks/tf-workshop.pdf

官方文档：
 1. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/how_tos/distributed/index.md
 2. https://www.tensorflow.org/versions/r0.10/how_tos/distributed/index.html

### notes about official document
```
# Start a TensorFlow server as a single-process "cluster".
$ python
>>> import tensorflow as tf
>>> c = tf.constant("Hello, distributed TensorFlow!")
>>> server = tf.train.Server.create_local_server()
>>> sess = tf.Session(server.target)  # Create a session on the server.
>>> sess.run(c)
'Hello, distributed TensorFlow!'
```
[tf.train.Server.create_local_server](https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html#Server.create_local_server)method creates a single-process cluster, with an in-process server.

### note about wangyi's reference

#### Prepare your cluster

###### Create a NFS Server in your cluster
Tensorflow requires a distributed file system to checkpoint images.任何文件系统都可以挂载到pods上。
kubernetes的pods中提到文件系统类型(ext4, xfs, etc) 

###### Create a NFS Server in your cluster
要有两个目录/data，/output。
###### Load the workshop data onto your NFS Server
使用kubectl create -f k8s-configs/load_data.yaml加载workshop中的数据到NFS服务器中的/data目录
###### Create a Tensorboard Server
Tensorboard Server is to display job logs, it will watch the /output目录 for events written by TensorFlow.
```
kubectl create -f k8s-configs/tensorboard.yaml
kubectl describe services tensorboard
```



#### Using the TensorFlow distributed runtime with Kubernetes


kubernetes支持cephfs(http://kubernetes.io/docs/user-guide/volumes/)










## Reference
 - [Tensorflow whitepaper](http://download.tensorflow.org/paper/whitepaper2015.pdf)
 
