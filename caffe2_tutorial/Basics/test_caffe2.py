from caffe2.python import workspace, model_helper
import numpy as np
import glog as log

# Create random tensor of three dimensions
x = np.random.rand(4, 3, 2)
print(x)
print(x.shape)

workspace.FeedBlob("my_x", x)

x2 = workspace.FetchBlob("my_x")
print(x2)

### Nets and Operators
# Create the input data
data = np.random.rand(16, 100).astype(np.float32)

# Create labels for the data as integers [0, 9].
label = (np.random.rand(16) * 10).astype(np.int32)

workspace.FeedBlob("data", data)
workspace.FeedBlob("label", label)

# Create model using a model helper
m = model_helper.ModelHelper(name="my first net")

weight = m.param_init_net.XavierFill([], 'fc_w', shape=[10, 100])
bias = m.param_init_net.ConstantFill([], 'fc_b', shape=[10, ])

fc_1 = m.net.FC(["data", "fc_w", "fc_b"], "fc1")
pred = m.net.Sigmoid(fc_1, "pred")
[softmax, loss] = m.net.SoftmaxWithLoss([pred, "label"], ["softmax", "loss"])

print(str(m.net.Proto()))

### Executing
# 1. initialization
m.AddGradientOperators([loss])
workspace.RunNetOnce(m.param_init_net)
# 2. create the actual training
workspace.CreateNet(m.net)
# 3. Run it
# Run 100 x 10 iterations
for j in range(0, 100):
  data = np.random.rand(16, 100).astype(np.float32)
  label = (np.random.rand(16) * 10).astype(np.int32)

  workspace.FeedBlob("data", data)
  workspace.FeedBlob("label", label)

  workspace.RunNet(m.name, 10)   # run for 10 times

# print(workspace.FetchBlob("softmax"))
log.info('The loss of forward running: %f' % workspace.FetchBlob("loss"))

print(str(m.net.Proto()))
