import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Created 3D plot of trajectories")
	parser.add_argument('-i', required=True, type=str, dest='filename', help='Input track position file')
	options = parser.parse_args()

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	fig.hold(True)
	
	track_data = np.genfromtxt(options.filename, delimiter=',')
	tracks = np.unique(track_data[:,0])

	for t in tracks:
		rows = track_data[track_data[:,0]==t]
		ax.plot(rows[:,1], rows[:,2], rows[:,4])
		ax.scatter(rows[0,1], rows[0,2], rows[0,4])

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Time')

	plt.show()

