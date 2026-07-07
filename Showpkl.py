import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the .pkl file
with open("Object1/features/photo_2024-04-08_14-09-00 (2).pkl", "rb") as f:
    temp_array = pickle.load(f)

# Assuming each float in temp_array represents a coordinate
# Create separate lists for x, y, and z coordinates
x = [point for point in temp_array[::3]]  # Extract every third element as x
y = [point for point in temp_array[1::3]]  # Extract every third element starting from index 1 as y
z = [point for point in temp_array[2::3]]  # Extract every third element starting from index 2 as z

# Plot 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Points')

plt.show()

plt.show()

