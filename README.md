# learn-tensorflow
Build a Distributed Tensorflow System

http://events.linuxfoundation.org/sites/events/files/slides/Container%20%2B%20DL-zhaojp-emc-v3.pdf

https://github.com/amygdala/tensorflow-workshop/tree/master/workshop_sections/distributed_tensorflow

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/dist_test

## 单机GPU测试

利用[矩阵运算测试](https://github.com/k8sp/k8s-tensorflow/pull/16)
```
镜像源FROM b.gcr.io/tensorflow/tensorflow:latest-gpu
docker build -t gpu_docker_tf_mat_multi .
docker run -it  --privileged -v /var/lib/nvidia:/usr/local/nvidia/lib64 gpu_docker_tf_mat_multi
nvidia驱动目录/var/lib/nvidia mount到/usr/local/nvidia/lib64 实现共享
```


##OpenAi
Infrastructure for Deep Learning

https://openai.com/blog/infrastructure-for-deep-learning/?utm_source=wanqu.co&utm_campaign=Wanqu+Daily&utm_medium=social


## TensorFlow tutorials

tensorflow将计算表示为图，图中的节点被称作ops。











