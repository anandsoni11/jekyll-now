---
layout: post
title: Getting Started with TensorFlow
category: binary
---

TensorFlow is an open source machine intelligence library developed at Google. It uses data flow graphs to do numerical computation. A few months back, I started to explore TensorFlow. When good ideas from academia and high-performance distributed computing are implemented in a promising tool like TensorFlow, it bodes well for the future! In this post, I will just attempt to describe some TensorFlow basics and give an insight into how a basic TF code looks like. ***Most*** of this information and sample code is also available on the official TensorFlow documentation.

### **Feed the programmer within you first!**
Google open sourced TensorFlow an year ago! Have a look : [TensorFlow](https://github.com/tensorflow/tensorflow)

If you have the inclination for research, this white paper might feed you more than code : [TensorFlow Whitepaper](http://download.tensorflow.org/paper/whitepaper2015.pdf)

### **Hello, World! with TensorFlow**
```
$ python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, World!')
>>> sess = tf.Session()
>>> sess.run(hello)
Hello, World!
```

### **Setting It Up**	
Setting TensorFlow up is easy. Follow the official guide : [TensorFlow Setup](https://www.tensorflow.org/get_started/os_setup).

However, an important thing to note here is that the TensorFlow Python API supports Python 2.7 and Python 3.3+. Also, since TensorFlow supports distributed computation on multiple CPUs/GPUs, the GPU version works best with Cuda Toolkit 8.0 and cuDNN v5.1. More on this can be found on the official documentation.

### **TensorFlow : Basic Usage** 
As described in the official documentation, TensorFlow uses data flow graphs to represent and do calculations. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. Nodes in the graph are called *ops* (short for operations). An *op* takes zero or more *Tensors*, performs some computation, and produces zero or more *Tensors*. In TensorFlow terminology, a *Tensor* is a typed multi-dimensional array. 

TensorFlow usage involves a few basics to be understood first.
#### **Variables**
Variables in TensorFlow are nothing but buffers that store tensors in memory. The *tf.Variable* class provides instantiation for a variable.
Sample code :

```
import tensorflow as tf

# Create a variable.
w = tf.Variable(<initial-value>, name=<optional-name>)
``` 
We can note from the above that a TF variable must be initialized. The Variable() constructor requires an initial value for the variable, which can be a Tensor of any type and shape. More details on variables can be found here : [TF Variables](https://www.tensorflow.org/how_tos/variables/).

#### **Sessions**
A Session places the graph ops onto Devices, such as CPUs or GPUs, and provides methods to execute them. These methods return tensors produced by ops as numpy ndarray objects in Python, and as tensorflow::Tensor instances in C and C++.

#### **The computation graph**
A TensorFlow graph is a description of computations. Programs in TensorFlow are usually structured into two phases - a construction phase that builds a graph and an execution phase that utilizes *Session* to execute *ops* in the graph. Sample Code :

**Building the graph :**

```
import tensorflow as tf

# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)
```
The default graph now has three nodes: two constant() ops and one matmul() op. To actually multiply the matrices, and get the result of the multiplication, you must launch the graph in a session.

**Execution using sessions :**

```
# Launch the default graph.
sess = tf.Session()

# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# The call 'run(product)' thus causes the execution of three ops in the
# graph: the two constants and matmul.
#
# The output of the matmul is returned in 'result' as a numpy `ndarray` object.
result = sess.run(product)
print(result)
# ==> [[ 12.]]

# Close the Session when we're done.
sess.close()
```


#### **Tensors**
TensorFlow programs use a tensor data structure to represent all data -- only tensors are passed between operations in the computation graph. You can think of a TensorFlow tensor as an n-dimensional array or list. A tensor has a static type, a rank, and a shape.

### **TensorFlow : How computation happens on Devices**
As per the official documentation :

> The TensorFlow implementation translates the graph definition into executable operations distributed across available compute resources, such as the CPU or one of your computer's GPU cards. In general you do not have to specify CPUs or GPUs explicitly. TensorFlow uses your first GPU, if you have one, for as many operations as possible.

The abstraction of computation graph is what makes (of a few other things) TF a really powerful library. Scalability across distributed resources makes TF all the more promising. TF appears to be using the available resources as greedily as possible without the user having to manage CPU/GPU devices herself.

However, TF allows the user to choose the device on his own for a particular computation :

```
with tf.Session() as sess:
  with tf.device("/gpu:1"):
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmul(matrix1, matrix2)
    ...
```
Here, the user has more than one GPUs on her machine and wants to use GPU-1 for computation.

### **Closure**
There are a lot of concepts and capabilities that I plan to cover in the future posts (as I learn more). I am going to implement a few algorithms/models myself (using TF) which I will open source and write about soon. Additionally, I am also going to write about how TF compares, in general, to other similar libraries like Theano. Overall, TF is a well built and promising library. Personally, I feel that, Google open sourcing TF, will lead to a lot of new interest in practical machine learning. Traditions of *doing* machine learning might change with TF and when that happens, we all shall be there to witness. There is a lot of excitement as, within an year, TF has seen great adoption.
