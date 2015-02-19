import sys
import glob
import os
import ntpath
import numpy
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # get the arguments
    arguments = sys.argv
    if len(arguments) < 3:
        print "usage: {} <input_folder> <output_folder> [<max id>]".format(sys.argv[0])
        sys.exit(-1)
    if len(arguments) == 3:
        normalize = False
    else:
        normalize = sys.argv[3]
    input_folder = sys.argv[1].rstrip('/')
    output_folder = sys.argv[2].rstrip('/')
    # loop over all *.tif files
    for filename in glob.glob("{}/*.tif".format(sys.argv[1])):
        filename_new = "{}/{}.png".format(
            output_folder,
            os.path.splitext(ntpath.basename(filename))[0])
        print "Convert {} to {}".format(filename, filename_new)
        # read the file
        image = plt.imread(filename).astype(numpy.double)
        image = numpy.flipud(image)
        if not normalize:
            if image.max() != 0:
                normalize = image.max()
            else:
                normalize = 1.0
        # save with new colormap
        plt.imsave(filename_new, image[:,:,0], vmin=0, vmax=normalize)
    
