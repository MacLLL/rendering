import numpy as np

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def random_points(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    return vec

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# phi = np.linspace(0, np.pi, 20)
# theta = np.linspace(0, 2 * np.pi, 40)
# x = np.outer(np.sin(theta), np.cos(phi))
# y = np.outer(np.sin(theta), np.sin(phi))
# z = np.outer(np.cos(theta), np.ones_like(phi))

points = random_points(500)
# print(points.shape)
# print(points[0])
# print(points[1])
# print(points[2])
# print(np.linalg.norm(points, axis=0))
# print(np.sqrt(points[0][0]**2+points[1][0]**2+points[2][0]**2))


# norm = np.sqrt((points[0]-1)**2+(points[1]-1)**2+(points[2]-1)**2)
# norm = np.power(points[0]**2+points[1]**2+points[2]**2,1/2)
# points /= norm


fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
# ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
# ax.scatter(points[0], points[1], points[2], s=10, c='r', zorder=10)

#todo: along the x axis, move 1,
# ax.scatter(points[0]-10, points[1], points[2], s=10, c='b', zorder=10)

#todo: shpere
norm = np.power(points[0]**2+points[1]**2+points[2]**2,1/2)
scale=1
points_new = points * scale / norm
print(points_new)
# sphere
ax.scatter(points_new[0], points_new[1], points_new[2], s=10, c='g', zorder=10)
# move 10 align the y axis
ax.scatter(points_new[0], points_new[1]-10, points_new[2], s=10, c='r', zorder=10)



# 90 clockwise, (x,y,z) -> (y,-x,z)
ax.scatter(points_new[1]-10, -1*points_new[0], points_new[2], s=10, c='y', zorder=10)


# 90 clockwise, (x,y,z) -> (y,-x,z)
M=np.matrix([[0,1,0],[1,0,0],[0,0,1]])
points_newnew = M@points_new

ax.scatter(points_newnew[0]-10, points_newnew[1], points_newnew[2], s=10, c='black', zorder=10)

homo=np.ones((1,500))

points = np.concatenate((points_new,homo),axis=0)

M_in = np.matrix([[10,0,0,0],[0,10,0,0],[0,0,1,0]])
M_ext = np.matrix([[-1,0,0,10],[0,1,0,10],[0,0,1,10],[0,0,0,1]])

points = M_in @ M_ext @ points

# ax.scatter(points[0], points[1], points[2], s=10, c='red', zorder=10)

#camera
# ax.scatter(10, 10, 10, s=100, c='blue', zorder=10)

points /= points[2]

print(points)
# projected image
ax.scatter(points[0], points[1], s=10, c='orange', zorder=10)




# n = 2000

# x1, x2 = 20, 40
# y1, y2 = 10, 20
# z1, z2 = 25, 50
#
# xs = (x2 - x1)*points[0] + x1
# ys = (y2 - y1)*points[1] + y1
# zs = (z2 - z1)*points[2] + z1
#
# ax.scatter(xs, ys, zs, s=10, c='orange', zorder=10)
plt.show()


'''
(x**2+y**2+z**2)**2 = 
'''


