# Pytorch Survey
- 关联参考
    - https://github.com/PaddlePaddle/Paddle/wiki/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%B1%82%E4%B8%8E%E6%93%8D%E4%BD%9C%E6%B3%A8%E5%86%8C%E6%9C%BA%E5%88%B6%E8%B0%83%E7%A0%94#pytorch
    - https://github.com/wangkuiyi/deeplearning/wiki/PyTorch
## 特点

1. 纯 Python 包，不是 C/C++ 到 Python 的绑定，和 numpy 无缝结合，立即执行
    - 后端核心代码用 C / C++ extending the Python interpreter 的方式写成: https://github.com/pytorch/pytorch/tree/master/torch/csrc
    - Pytorch 是一个Python 包，可以做到“写一行代码，立即执行”
    - 使用 Pytroch 是用 Python 语言在写程序，而不是“写配置”
    - 神经网络相关的抽象，都用 Python 重写过 （https://github.com/pytorch/pytorch/tree/master/torch 下 nn， optim 等），因此，想要扩展 Pytorch 是非常容易的，只要继承`nn.Module` （神经网络的基类），在此基础上扩展
    - 直接用 Python 的条件控制语句即可实现控制流的改变，实现动态图非常方便，不需要 tensorflow 中[control flow operation](https://www.tensorflow.org/api_guides/python/control_flow_ops) 的概念

2. 动态构建计算图
    - 这一特性源自于 Pytroch 就是一个 Python 包，Python 是一种解释型语言
    - iteration（一个batch）之间计算图结构可以改变，但一个 iteration 之的内多条训练数据共享同一个计算图
        - [TensorFlow Fold](https://research.googleblog.com/2017/02/announcing-tensorflow-fold-deep.html) (deeppaper: [Deep Learning with Dynamic Computation Graphs](https://arxiv.org/abs/1702.02181)) 通过 dynamic batching 算法实现了 一个 iteration 之内多条样本计算图都不相同的数据并行，是一种用“静态”模拟“动态”的算法（论文细节没有细看）
        - 动态图的特性[令 recursive network 实现变得简单](https://devblogs.nvidia.com/parallelforall/recursive-neural-networks-pytorch/)，但是（例如 TreeLstm 只能以 batch size = 1 来计算），Pytorch 默认没有实现广义动态图（不局限于序列） batch 并行计算的机制，用户需要/可以自己来实现
    - 由于每个iteration 之内多条样本计算图结构不可改变，在 0.1.10 release 之前 RNN 并行计算逻辑需要自己实现。目前这一问题已经得到改善。
        - 0.1.10 release 之后的 packedsequence 实现了 RNN 的 batch 并行，http://pytorch.org/docs/nn.html#packedsequence ，原理和 PaddlePaddle 的无 Padding RNN 一样
        - https://github.com/pytorch/pytorch/releases/tag/v0.1.10
        - https://discuss.pytorch.org/t/about-the-variable-length-input-in-rnn-scenario/345
        - https://discuss.pytorch.org/t/how-to-do-mini-batch-with-dynamic-computation-graph/430
       
3. 基于反向模式自动微分（Reverse-mode auto-differentiation）技术的动态网络
    - 由于（1） Tensor 和基于Tensor的细粒度各种矩阵运算都直接暴露给用户；（2）自动求导机制，在**绝大多数**对神经网络功能的扩展中（比如定义新的网络，实现论文中一个新的Layer），都只需要写前向而不需要写反向，对研究者来说，可以节约大量的开发时间
    - 可以零延迟或零成本地任意改变网络的行为
    - 基于反向模式自动微分技术并非Pytroch独有，但 Pytroch 声称自己是到目前为止最快的实现
        - 参考：
            1. [autograd：HIPS/autograd](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md)
            2. [Chainer：A flexible framework for neural networks](https://www.slideshare.net/beam2d/introduction-to-chainer-a-flexible-framework-for-deep-learning)
    - backward of backward （Hessian Matrix of weights）
        - tensorflow 已经支持此功能，但是返回的是 Hessian-vector product，不是 Hessian Matrix (**?? 存疑**) 
        - 在下一个 release （0.2）中支持 https://discuss.pytorch.org/t/is-there-any-way-to-get-second-order-derivative-or-hessian-matrix/1149  ，返回 Hessian Matrix （**不完全确定？**） 
        - Hessian Matrix 基于自动求导机制按照微积分定理来产生，对与没有自动求导机制的框架不易以通用的方式实现 

4. 支持单机多卡，尚不支持多机多卡，开发中：https://github.com/pytorch/pytorch/issues/241

### Pytorch 暴露的一些抽象概念
1. Tensor：操作的基本数据
    - Tensor 之上定义的 operation 高度类似于 numpy 中的各种运算，细粒度的 operations 使得自定义神经网络非常方便
2. 自动求导相关
    - Variable ：是 Tensor 的一个wrapper
        - 不仅保存值，而且保存这个值的 creator，需要自动求导的子图必须使用 Variable 而不是 Tensor 参与运算
        - Variable 支持的 operation 和 Tensor 一致
    - Function
        - 记录 operation 历史（Function 对象的 DAG），定义求导 operation 的公式
            1. Variable 上的每一个 operation （对 Variable 内 wrap 的 Tensor 的各种运算）都会创建一个新的 Function对象，这个对象（1）完成计算；（2）记录计算的历史
            2. 计算的历史是以 Function 对象的 DAG 形式来记录，边是数据依赖（input <- output）
            3. 当 backward 方法被调用时，这个 DAG 按照从根向叶遍历的顺序调用每一个 Function 对象的 backward 方法，然后将返回结果传递给下一个（孩子结点）Function 对象，

3. Module：所有神经网络的基类

---
## Pytorch 对比 TensorFlow 使用体验的优点

这里只讨论PyTorch相较于其他框架，值得学习的优点。

### 几乎原生的python使用体验
直接构建自 Python C API，从细粒度上直接支持python的访问。

带来的优势：

- 完全 python 化的使用体验，降低 pythoner 适应的门槛
- 可以直接用原生python语法定义新的 operation

这些方面，tensorflow 完全是另外一种体验：

- 总有一种用 python 调用 C++ 写的第三方动态链接库的感觉
- 写模型需要更多代码，无法贯彻 python 简约风格
- 新的 operation 必须用 C++ 开发
    
下面演示 PyTorch 与 python 结合的紧密程度[1]：

比如用 numpy 实现一个 2 层的神经网络：

```python
import numpy as np
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
  # Forward pass: compute predicted y
  h = x.dot(w1)
  h_relu = np.maximum(h, 0)
  y_pred = h_relu.dot(w2)
  
  # Compute and print loss
  loss = np.square(y_pred - y).sum()
  print(t, loss)
  
  # Backprop to compute gradients of w1 and w2 with respect to loss
  grad_y_pred = 2.0 * (y_pred - y)
  grad_w2 = h_relu.T.dot(grad_y_pred)
  grad_h_relu = grad_y_pred.dot(w2.T)
  grad_h = grad_h_relu.copy()
  grad_h[h < 0] = 0
  grad_w1 = x.T.dot(grad_h)
 
  # Update weights
  w1 -= learning_rate * grad_w1
  w2 -= learning_rate * grad_w2
```
相同的功能使用 PyTorch 实现

```python
import torch

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in).type(dtype)
y = torch.randn(N, D_out).type(dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)

learning_rate = 1e-6
for t in range(500):
  # Forward pass: compute predicted y
  h = x.mm(w1)
  h_relu = h.clamp(min=0)
  y_pred = h_relu.mm(w2)

  # Compute and print loss
  loss = (y_pred - y).pow(2).sum()
  print(t, loss)

  # Backprop to compute gradients of w1 and w2 with respect to loss
  grad_y_pred = 2.0 * (y_pred - y)
  grad_w2 = h_relu.t().mm(grad_y_pred)
  grad_h_relu = grad_y_pred.mm(w2.t())
  grad_h = grad_h_relu.clone()
  grad_h[h < 0] = 0
  grad_w1 = x.t().mm(grad_h)

  # Update weights using gradient descent
  w1 -= learning_rate * grad_w1
  w2 -= learning_rate * grad_w2
```
除去几个激活函数，**PyTorch 在实现中几乎只引入了一个 `FloatTensor` 的概念，主要的代码结构和原生的 python 实现基本一致。**

再看看 tensorflow 的情况：

```python
import tensorflow as tf
import numpy as np

# First we set up the computational graph:

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create placeholders for the input and target data; these will be filled
# with real data when we execute the graph.
x = tf.placeholder(tf.float32, shape=(None, D_in))
y = tf.placeholder(tf.float32, shape=(None, D_out))

# Create Variables for the weights and initialize them with random data.
# A TensorFlow Variable persists its value across executions of the graph.
w1 = tf.Variable(tf.random_normal((D_in, H)))
w2 = tf.Variable(tf.random_normal((H, D_out)))

# Forward pass: Compute the predicted y using operations on TensorFlow Tensors.
# Note that this code does not actually perform any numeric operations; it
# merely sets up the computational graph that we will later execute.
h = tf.matmul(x, w1)
h_relu = tf.maximum(h, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)

# Compute loss using operations on TensorFlow Tensors
loss = tf.reduce_sum((y - y_pred) ** 2.0)

# Compute gradient of the loss with respect to w1 and w2.
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# Update the weights using gradient descent. To actually update the weights
# we need to evaluate new_w1 and new_w2 when executing the graph. Note that
# in TensorFlow the the act of updating the value of the weights is part of
# the computational graph; in PyTorch this happens outside the computational
# graph.
learning_rate = 1e-6
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# Now we have built our computational graph, so we enter a TensorFlow session to
# actually execute the graph.
with tf.Session() as sess:
  # Run the graph once to initialize the Variables w1 and w2.
  sess.run(tf.global_variables_initializer())

  # Create numpy arrays holding the actual data for the inputs x and targets y
  x_value = np.random.randn(N, D_in)
  y_value = np.random.randn(N, D_out)
  for _ in range(500):
    # Execute the graph many times. Each time it executes we want to bind
    # x_value to x and y_value to y, specified with the feed_dict argument.
    # Each time we execute the graph we want to compute the values for loss,
    # new_w1, and new_w2; the values of these Tensors are returned as numpy
    # arrays.
    loss_value, _, _ = sess.run([loss, new_w1, new_w2],
                                feed_dict={x: x_value, y: y_value})
    print(loss_value)
```
直观感觉 tensorflow 的实现要复杂一些：

- 引入了 `placeholder`, `Variable`, `tf.Session`, `feed_dict` 等新概念
    - 新手可能不清楚 `placeholder` 和 `Variable` 的概念
        - 进而要去了解 tensorflow 计算图模型，实现细节等
        - 有点反小白
- 代码多了很多，相当一部分跟原生代码的结构不太一致


<b><font color="red">用 python 扩展 PyTorch 的 operaion 也很很简单</font></b>：

```python
import torch
from torch.autograd import Variable

class MyReLU(torch.autograd.Function):
  """
  We can implement our own custom autograd Functions by subclassing
  torch.autograd.Function and implementing the forward and backward passes
  which operate on Tensors.
  """
  def forward(self, input):
    """
    In the forward pass we receive a Tensor containing the input and return a
    Tensor containing the output. You can cache arbitrary Tensors for use in the
    backward pass using the save_for_backward method.
    """
    self.save_for_backward(input)
    return input.clamp(min=0)

  def backward(self, grad_output):
    """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
    input, = self.saved_tensors
    grad_input = grad_output.clone()
    grad_input[input < 0] = 0
    return grad_input
```

综合上面原生python，PyTorch, tensorflow 对同一个模型实现的比较可以看出 PyTorch 的优势在以下几点：

- **相比于原生的实现，引入的新概念很少，降低了 python 用户理解的门槛**
- 代码基本跟原生的 python 实现一致，统一的 python 实现的思维
- **可以直接用原生 python 代码扩展 PyTorch 的 operation** 

### 基于 tensor 的扩展能力
这一点上 PyTorch 和 Tensorflow 类似，可以用 build-in 的 operation 去组合出新的大粒度的 operation；而对比 paddle/Caffe 粗粒度的layer 支持， PyTorch/Tensorflow 能真正实现NN编程，而前者更大程度上是在配置NN。
### API封装设计合理（与Keras/tensorflow对比）
这里我们只讨论 PyTorch 的原生API，不同于 Tensorflow 有 Keras, tf.contrib, tf.learn 等多套官方API，**PyTorch 官方只有一套 API，但良好的上手体验并不逊色于 Tensorflow 的几套 API。**

PyTorch的API有点类似 Keras 和 Tensorflow 裸API的结合：

- 常用的 NN 模块封装上类似 Keras 的 functional API，layer作为 function 对不同的 input 输出包装好的 output tensor
  - 比如 `conv1 = nn.Conv2d(1 10, kernal_size=5)`
- 除了对常用 NN 模块的封装，其他API没有做太多封装/修饰，保持底层和简洁（用起来类似 Tensorflow 裸 API）
    - Keras 中很多封装非常高层，但用户无法避免依赖 Tensorflow 的原生接口
        - 比如 `Model` 接口基本隐藏了底层 Tensorflow 的所有细节，但大多数情况，类似 `tf.scope`, `tf.device` 等原生接口的使用是无法避免的，这就需要两种层次的接口混用
        - Keras 和 Tensorflow 原生API的风格不太一致，增加了用户在两者切换/混用的学习门槛

总之，PyTorch的API设计上，粒度适中，并且官方的接口比较统一，方便用户持续积累好的代码模式；相比较 TensorFlow 则有多套接口，且抽象的程度各不相同，可能不利于用户经验的积累。

### Dynamic net相关
得益于直接基于 python C API 构建的 python 接口，PyTorch 支持动态网络的构建；不同于 Tensorflow 在运行前需要生成静态计算图，PyTorch的程序可以在执行时动态构建/调整计算图。

比如 TreeLSTM[\[2](#参考文献)\] 模型将 LSTM 构建在一个树结构的网络上，具体应用比如，以语法树为形状构建神经网络，用来学习句子信息；每个句子都有不同的语法树，因此需要构建不同的神经网络（树各个节点的模型参数共享，但连接关系会动态改变）。

这个模型 PyTorch 和 TensorFlow 都有支持，前者利用了自己支持 dynet 的优势，实现起来比较自然；后者通过`tf.while_loop` 等条件分支 operation 支持，需要对原始逻辑进一步抽象才能支持。

其中， PyTorch 的实现直接使用了递归的方式遍历树来构建网络（numpy 怎么实现，PyTorch就可以）[4]

```python
def expr_for_tree(self, tree, decorate=False):
    if tree.isleaf():
      return self.embed(makevar(self.w2i.get(tree.label, 0)))
    if len(tree.children) == 1:
      assert(tree.children[0].isleaf())
      expr = self.expr_for_tree(tree.children[0])
      if decorate:
        tree._e = expr
      return expr
    assert(len(tree.children) == 2), tree.children[0]
    e1 = self.expr_for_tree(tree.children[0], decorate)
    e2 = self.expr_for_tree(tree.children[1], decorate)
    expr = self.TANH(self.WR(torch.cat((e1, e2),1)))
    if decorate:
      tree._e = expr
    return expr
```

Tensorflow 无法支持上面递归编译树的方式，只能把树的遍历加工成类似 map() 的操作[5]，作者脑洞很大啊。

大体上，把二叉树用一个 `2XN` 的数组表示，比如

```
[ 
[-1, 2, 3],
[3, 1, 0]]
```
其中 `N` 表示树的节点数（需要把所有的树塞进同一个tensor中，因此shape必须取最大的），其中每一列的两个数字表示当前节点左右孩子的节点 id，排列按照从叶子节点往根节点的遍历次序。

如此，可以将这个数组放进类似 map 的模块中，部分代码如下：

```python
 def _recurrence(node_h,node_c,idx_var):
                node_info=tf.gather(treestr,idx_var)

                child_h=tf.gather(node_h,node_info)
                child_c=tf.gather(node_c,node_info)

                flat_ = tf.reshape(child_h,[-1])
                tmp=tf.matmul(tf.expand_dims(flat_,0),cW)
                u,o,i,fl,fr=tf.split(1,5,tmp)

                i=tf.nn.sigmoid(i+bi)
                o=tf.nn.sigmoid(o+bo)
                u=tf.nn.tanh(u+bu)
                fl=tf.nn.sigmoid(fl+bf)
                fr=tf.nn.sigmoid(fr+bf)

                f=tf.concat(0,[fl,fr])
                c = i * u + tf.reduce_sum(f*child_c,[0])
                h = o * tf.nn.tanh(c)

                node_h = tf.concat(0,[node_h,h])

                node_c = tf.concat(0,[node_c,c])

                idx_var=tf.add(idx_var,1)

                return node_h,node_c,idx_var
            loop_cond = lambda a1,b1,idx_var: tf.less(idx_var,n_inodes)

            loop_vars=[node_h,node_c,idx_var]
            node_h,node_c,idx_var=tf.while_loop(loop_cond, _recurrence,
                                                loop_vars,parallel_iterations=10)
```
上面实现中，对问题本身的抽象比较难，之后会用到 `tf.while_loop` 来支持类似 map 的扫描。

相比较，**对于动态网络的问题，PyTorch 从任务抽象到具体实现都要简单自然很多**。 


### 参考文献
1. http://icode.baidu.com/repo/baidu%2Fgtm%2Fgtmlib2/files/master/tree/
2. Improved Semantic Representations From
Tree-Structured Long Short-Term Memory Networks
3. https://www.tensorflow.org/api_docs/python/tf/while_loop
4. [PyTorch implementation for TreeLSTM](https://gist.github.com/wolet/1b49c03968b2c83897a4a15c78980b18)
5. [Tensorflow implementation for TreeLSTM](https://gist.github.com/wolet/1b49c03968b2c83897a4a15c78980b18)
