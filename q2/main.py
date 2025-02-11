#!/usr/bin/env python

from os.path import isdir
from os import mkdir

from numpy import array, meshgrid, arange, mean
from matplotlib import pyplot as plt

data_dir = "Local_density_of_states_near_band_edge"
if not isdir(data_dir):
	print("Downloading data...")
	from urllib.request import urlretrieve
	mkdir(data_dir)
	for i in range(11):
		urlretrieve(f"https://raw.githubusercontent.com/Physics-129AL/Local_density_of_states_near_band_edge/refs/heads/main/local_density_of_states_for_level_{i}.txt", f"{data_dir}/{i}.txt")

def loadtxt(filename):
	with open(filename) as f:
		return array([list(map(float, line.split(", "))) for line in f])

# a
heatmap_dir = "local_density_of_states_heatmap"
if not isdir(heatmap_dir):
	print("Generating heatmaps...")
	mkdir(heatmap_dir)
	for i in range(11):
		data = loadtxt(f"{data_dir}/{i}.txt")
		plt.imshow(data, cmap="hot")
		plt.colorbar()
		plt.title(f"Local Density of States for Level {i}")
		plt.savefig(f"{heatmap_dir}/{i}.png")
		plt.close()

# b
height_dir = "local_density_of_states_height"
if not isdir(height_dir):
	print("Generating height plots...")
	mkdir(height_dir)
	for i in range(11):
		data = loadtxt(f"{data_dir}/{i}.txt")
		x, y = meshgrid(arange(data.shape[1]), arange(data.shape[0]))
		fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
		ax.plot_surface(x, y, data, cmap="hot")
		ax.set_title(f"Local Density of States for Level {i}")
		plt.savefig(f"{height_dir}/{i}.png")
		plt.close()

# c
def subregion(data):
	return data[180:220, 100:130] # some predefined subregion
averages = [mean(subregion(loadtxt(f"{data_dir}/{i}.txt"))) for i in range(11)]
plt.plot(averages)
plt.xlabel("Level")
plt.ylabel("Average Local Density of States in a Subregion")
plt.show()
# The plot does not seem to show any clear trend.
