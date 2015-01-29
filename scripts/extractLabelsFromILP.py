import vigra
import numpy as np
import h5py
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Extract and export the user specified pixel classification labels from an ilastik project file")
	parser.add_argument('--shape', metavar='shape', required=True, type=int, nargs='+', help='Shape of each image given as 2 or 3 ints (time axes is handled internally)')
	parser.add_argument('--project', dest='project', required=True, type=str, help='Filename of ilastik project')
	parser.add_argument('--out_prefix', dest='out_prefix', required=True, type=str, help='Prefix (path/fileprefix) that will be used for all output files, appended by the frame number and the extension png')
	args = parser.parse_args()

	in_f = h5py.File(args.project, 'r')
	imageShape = tuple(args.shape)
	out_prefix = args.out_prefix

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

		# save image if there is not just label 0 inside
		if len(np.unique(fullLabelFrame)) > 1:
			vigra.impex.writeImage(fullLabelFrame, str(out_prefix + frameNum + '.png'))