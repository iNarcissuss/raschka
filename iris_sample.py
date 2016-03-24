import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from Perceptron import Perceptron

# Download and parse the iris dataset from online
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# iloc is integer location, and is how you can index the dataframe 
# using integer positions
# "0:100" says to get rows 0 to 100 (exclusive)
# "4" says to return only the 4th column
# In the iris dataset, the 4th column is the classification of the data
# calling "values" returns only the classification (aka column 4) instead of
# both the row number and the classification
# Coincidentally, there's 50 Iris-setosa and 50 Iris-versicolor samples
# This is why we have 0:100, so that we can get half and half of each
y = df.iloc[0:100, 4].values

# Takes all of the classifications in y and replaces them with numeric values
# -1 for if the classification is 'Iris-setosa' and 1 if the classification
# is 'Iris-versicolor'
y = np.where(y == 'Iris-setosa', -1, 1)

# For the first 100 columns we grab columns 0 and 2 (the first and third columns)
# The first column represents the sepal length of the flower
# Sepals are the "leaves" that grow underneath the petal
# The third column represents the petal length of the flower
# Petals are the usually colorful and softer parts of a flower
# We are going to use only the sepal length and petal length to classify the flowers
X = df.iloc[0:100, [0, 2]].values

# Plot the setosa samples using sepal length on the X axis
# and petal length on the Y axis.
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')

# Same as plotting setosa, but this time use blue and 'x's instead of 'o's
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

# Set labels
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')

# on Karma-Ubuntu the backend for matplotlib needed to be changed to be Qt4Agg
# instead of agg
# On Karma-Ubuntu this can be changed in the following file:
# sudo vi ~/ml/raschka/venv/lib/python2.7/site-packages/matplotlib/mpl-data/matplotlibrc
plt.show()

# Create a perceptron with learning reate 0.1 and 10 iterations
ppn = Perceptron(eta=0.1, n_iter=10)
# Call the function that should fit a linear regression
ppn.fit(X, y)
# Plot with the number of iterations (also called epochs) on the x axis
# on the y axis we have the number of mistakes made during that iteration
plt.plot(range(1, ppn.n_iter + 1), ppn.errors_, marker='o')

# Set labels
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

def plot_decision_regions(X, y, classifier, resolution=0.02):
  # setup marker generator and color map
  # markers = square, x, o, triangle pointing up, triangle pointing down
  markers = ('s', 'x', 'o', '^', 'v')
  # color options for up to 5 types of data
  colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
  # Creates a color map taking a subset of "colors" based on the number of types
  # of data in y (which are the classifications)
  cmap = ListedColormap(colors[:len(np.unique(y))])

  # plot the decision surface  
  # x1_min anx x1_max represent the smallest and largest values on the x axis
  # subtract 1 and add 1 to the min and max so that there's padding
  x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  # x2 represents the y axis
  x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

  # Color the grids 
  xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
  Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
  Z = Z.reshape(xx1.shape)
  plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
  plt.xlim(xx1.min(), xx1.max())
  plt.ylim(xx2.min(), xx2.max())

  # Loops through the number of classifications and plots and colors everything one
  # classification at a time
  for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
