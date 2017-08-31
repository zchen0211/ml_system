import mxnet as mx

a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
c = a + b
print (a, b, c)

# elemental wise multiplication
d = a * b
# matrix multiplication
e = mx.sym.dot(a, b)
# reshape
f = mx.sym.reshape(d+e, shape=(1,4))
# broadcast
g = mx.sym.broadcast_to(f, shape=(2,4))
# plot
mx.viz.plot_network(symbol=g)
