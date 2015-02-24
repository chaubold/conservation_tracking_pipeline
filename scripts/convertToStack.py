import vigra
import numpy as np
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Create a hdf5 stack from a series of tif images")
	parser.add_argument('--prefix', dest='prefix', required=True, type=str, help='Prefix of filename before number component')
	parser.add_argument('--extension', dest='extension', default='tif', type=str, help='Extension of filename (default = tif)')
	parser.add_argument('--max-frame', dest='max_frame', type=int, required=True, help='Number of last frame to take into account (starts at 0, inclusive)')
	parser.add_argument('--dims', dest='dims', type=int, default=2, help='Number of dimensions of the data, default=2')
	parser.add_argument('--out_name', dest='out_name', default='stack.h5', type=str, help='Filename of output stack, default=stack.h5')
	args = parser.parse_args()
	
	name_prefix = args.prefix
	extension = '.' + args.extension
	
	max_frame_num = args.max_frame

	if args.dims == 2:
		loadFunc = vigra.impex.readImage
	else:
		loadFunc = vigra.impex.readVolume

	d = loadFunc(name_prefix+'{:03}'.format(0)+extension)
	d = np.expand_dims(d, axis=0)

	for i in range(1,max_frame_num+1):
	    slice = loadFunc(name_prefix+'{:03}'.format(i)+extension)
	    slice = np.expand_dims(slice, axis=0)
	    d = np.append(d, slice, axis=0)
	    print("Added file: {}{:03}{}".format(name_prefix, i, extension))

	vigra.impex.writeHDF5(d, args.out_name, 'data', compression='gzip')
	print("Saved stack to {}".format(args.out_name))