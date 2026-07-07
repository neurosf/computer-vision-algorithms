
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

# Load .ply file
mesh = o3d.io.read_triangle_mesh("point_cloud/sparseObj1.ply")
vertices = np.array(mesh.vertices)
triangles = np.array(mesh.triangles)

# Plot using matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot vertices
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='o')

# Plot triangles
for triangle in triangles:
    v0 = vertices[triangle[0]]
    v1 = vertices[triangle[1]]
    v2 = vertices[triangle[2]]
    ax.plot([v0[0], v1[0], v2[0], v0[0]], 
            [v0[1], v1[1], v2[1], v0[1]], 
            [v0[2], v1[2], v2[2], v0[2]], 'r')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
