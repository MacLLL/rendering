import numpy as np
import matplotlib.pyplot as plt


sdf = np.load('/home/luy100/projects/CAR/NeuS/graphs/sdf.npy')
alpha = np.load('/home/luy100/projects/CAR/NeuS/graphs/alpha.npy')
weight = np.load('/home/luy100/projects/CAR/NeuS/graphs/weights.npy')

sdf=sdf[:,64:]
alpha=alpha[:,64:]
weight=weight[:,64:]
# print(sdf)

x = np.linspace(0, 64, 64)

print(sdf.shape)

ray1 = sdf[:1,:].reshape(-1,1)
ray2 = sdf[1:2,:].reshape(-1,1)
ray3 = sdf[2:3,:].reshape(-1,1)
ray4 = sdf[3:4,:].reshape(-1,1)

plt.plot(x, ray1)
plt.plot(x, ray2)
plt.plot(x, ray3)
plt.plot(x, ray4)

# plt.plot(ypoints, linestyle = 'dotted')
plt.show()


ray1 = alpha[:1,:].reshape(-1,1)
ray2 = alpha[1:2,:].reshape(-1,1)
ray3 = alpha[2:3,:].reshape(-1,1)
ray4 = alpha[3:4,:].reshape(-1,1)

plt.plot(x, ray1)
plt.plot(x, ray2)
plt.plot(x, ray3)
plt.plot(x, ray4)

# plt.plot(ypoints, linestyle = 'dotted')
plt.show()

ray1 = weight[:1,:].reshape(-1,1)
ray2 = weight[1:2,:].reshape(-1,1)
ray3 = weight[2:3,:].reshape(-1,1)
ray4 = weight[3:4,:].reshape(-1,1)

plt.plot(x, ray1)
plt.plot(x, ray2)
plt.plot(x, ray3)
plt.plot(x, ray4)

# plt.plot(ypoints, linestyle = 'dotted')
plt.show()