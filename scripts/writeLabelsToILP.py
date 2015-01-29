import vigra
import numpy as np
import h5py
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Use all pixels != 0 in the given image as label for the specified class, of the specified frame in the dataset")
	#parser.add_argument('--shape', metavar='shape', required=True, type=int, nargs='+', help='Shape of each image given as 2 or 3 ints (time axes is handled internally)')
	parser.add_argument('--project', dest='project', required=True, type=str, help='Filename of ilastik project')
	parser.add_argument('--labels', dest='labelFile', required=True, type=str, help='file from which to take labels')
	parser.add_argument('--frame', dest='frameNo', required=True, type=int, help='timeframe number')
	parser.add_argument('--class', dest='labelClass', required=True, type=int, help='class to assign these labels to')
	args = parser.parse_args()

	in_f = h5py.File(args.project, 'r+')
	labelImage = vigra.impex.readImage(args.labelFile)
	labelImage = labelImage.squeeze()
	labelImage[labelImage > 0] = args.labelClass

	if len(labelImage.shape) == 2:
		blockSlice = "[0:{},0:{},0:1]".format(labelImage.shape[0], labelImage.shape[1])
	elif len(labelImage.shape) == 3:
		blockSlice = "[0:{},0:{},0:{},0:1]".format(labelImage.shape[0], labelImage.shape[1], labelImage.shape[2])
	else:
		raise Exception("Unsupported Label image shape")

	try:
	    labels = in_f['/PixelClassification/LabelSets']
	except:
	    raise Exception("No labeling datasets in provided HDF5-File")

	# find frame that we want to add labels to
	frame = 'labels{:03}'.format(args.frameNo)
	assert(frame in labels.keys())

	frameLabelBlockList = labels[frame]

	if len(frameLabelBlockList.keys()) > 0:
		print("Warning: overwriting all previously specified labels")
		for i in frameLabelBlockList.keys():
			del frameLabelBlockList[i]

	blockName = 'block0000'
	frameLabelBlockList.create_dataset(blockName, data=labelImage, dtype='u8')
	frameLabelBlockList[blockName].attrs['blockSlice'] = blockSlice