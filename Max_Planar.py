import matplotlib.pyplot as plt

vertices = [15,17,17,22,36,47,40,48,30,41,35,18]
Percent = [95, 89.29, 75, 88.57, 86.76, 73.81, 75, 77.77, 66.66, 79.49, 76, 92.86]

# Create scatter plot
plt.scatter(vertices, Percent, color='blue', marker='o')  # optional: color and marker style
    
# Add labels and title
plt.xlabel("Number of Vertices")
plt.ylabel("Inner Edges (%)")
plt.title("Percent of Inner Edges Present vs Graph Size")
    
# Show the plot
plt.show()