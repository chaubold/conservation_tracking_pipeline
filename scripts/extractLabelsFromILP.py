import vigra
import numpy as np
import h5py

in_f = h5py.File('/Users/chaubold/Desktop/01_pixel_classification.ilp', 'r')
imageShape = (535,505)
out_prefix = '/Users/chaubold/Desktop/labels_'

try:
    labels = in_f['/PixelClassification/LabelSets']
except:
    print "No labeling datasets in provided HDF5-File"

for frame in labels.keys():
	frameLabels = labels[frame]
	fullLabelFrame = np.zeros(imageShape + (1,))
	frameNum = frame[-3:]

	for blockId in frameLabels.keys():
		block = frameLabels[blockId]
		if 'blockSlice' in block.attrs.keys():
			b = block.attrs['blockSlice']
			b = b.replace('[','')
			b = b.replace(']','')
			b = b.split(',')
			blockSlice = [[int(i) for i in s.split(':')] for s in b]

			if len(imageShape) == 2:
				fullLabelFrame[blockSlice[0][0]:blockSlice[0][1], blockSlice[1][0]:blockSlice[1][1], blockSlice[2][0]:blockSlice[2][1]] = block.value
			elif len(imageShape) == 3:
				fullLabelFrame[blockSlice[0][0]:blockSlice[0][1], blockSlice[1][0]:blockSlice[1][1], blockSlice[2][0]:blockSlice[2][1], blockSlice[3][0]:blockSlice[3][1]] = block.value

	vigra.impex.writeImage(fullLabelFrame, str(out_prefix + frameNum + '.png'))