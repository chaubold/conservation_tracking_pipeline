import sys
import glob
import os
import ntpath
import numpy
import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # shuffle
    numpy.random.seed()
    # get the arguments
    arguments = sys.argv
    if len(arguments) < 4:
        print "usage: {} <input_folder> <output_folder> <max id>".format(sys.argv[0])
        sys.exit(-1)
    max_id = int(sys.argv[3])
    input_folder = sys.argv[1].rstrip('/')
    output_folder = sys.argv[2].rstrip('/')
    # get the remapping
    #permutation = numpy.zeros(max_id + 1).astype(numpy.uint8)
    #permutation[1:] = numpy.random.permutation(max_id) + 1
    cmap_data = numpy.random.rand ( max_id, 3)
    cmap_data[0,:] = numpy.array([0,0,0])
    cmap = matplotlib.colors.ListedColormap ( cmap_data )
    # loop over all *.tif files
    for filename in glob.glob("{}/*.tif".format(sys.argv[1])):
        filename_new = "{}/{}.png".format(
            output_folder,
            os.path.splitext(ntpath.basename(filename))[0])
        print "Convert {} to {}".format(filename, filename_new)
        # read the file
        image = plt.imread(filename).astype(numpy.uint8)
        #image = vigra.imprex.readImage(filename).astype(numpy.uint8)[:,:,0]
        image = numpy.flipud(image)
        #for n in range(1,image.max() + 1, 1):
        #    image[image == n] = permutation[n]
        
        # save with new colormap
        plt.imsave(filename_new, image, vmin=0, vmax=max_id, cmap=cmap)
    
