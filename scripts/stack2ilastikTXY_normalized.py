#!/usr/bin/env python
# written by Martin Schiegg

'''Convert time-lapse image stack to ilastik h5 file.'''

import h5py
import vigra
import sys
import os
import os.path
import numpy as np
import optparse
import glob

if __name__ == '__main__':
    usage = """%prog [options] [multipage-]tifs"""
    parser = optparse.OptionParser(usage=usage)

    parser.add_option('--min', type='int', dest='min', default=0, help='set min for normalization from 16bit to 8bit [default: %default]')
    parser.add_option('--max', type='int', dest='max', default=-1, help='set max for normalization from 16bit to 8bit [default: %default]')
    parser.add_option('--determine-only', action='store_true', dest='determine', help='determine min and max for normalization purposes')
    parser.add_option('--end', type='int', dest='end', default=-1, help='last index of files included [default: %default]')

    options, args = parser.parse_args()

    numArgs = len(args)
    if numArgs > 0:
        in_tiffs = [] 
        for arg in args:
            in_tiffs.extend(glob.glob(arg))
        in_tiffs.sort()
    else:
        parser.print_help()
        sys.exit(1)
#    if len(sys.argv) == 1:
#        print "Arguments: multipage-tiffs"
#        sys.exit(1)

  
    ds = None
    dataType = 'UINT8'
    npType = np.uint8
    maxValue = 255

    MAX = float(options.max)
    MIN = float(options.min)

    print 'in_tiffs[0] = ', in_tiffs[0]
    nslices = vigra.impex.numberImages(in_tiffs[0])
    print 'nslices = ', nslices
    found = False
    if options.max == -1:
       for t in in_tiffs:
          for i in range(nslices):
             if np.max(vigra.readImage(t, index=i, dtype='UINT16')) > 255:
                 print 'at least one image is uint16'
                 dataType = 'UINT16'
                 npType = np.uint16
                 maxValue = 2.0**16-1.0
                 found = True
                 break
          if found:
             break
       if not found:
           print 'the images are all uint8, no normalization needed'
           npType = np.uint8
           MAX = 255
           MIN = 0
    
       if npType == np.uint16:
             MAX = 0
             MIN = 1000000
             for idx,t in enumerate(in_tiffs):
                 nslices = vigra.impex.numberImages(t)
                 img = [vigra.readImage( t , dtype=dataType, index=i ) for i in range(nslices)]
                 img = np.dstack(img)
                 #img = vigra.readImage(t, dtype=dataType, index=0)
                 #print 'max = ', np.max(img)
                 tmpMax = np.max(img)
                 tmpMin = np.min(img)
                 if tmpMax > MAX:
                     MAX = tmpMax
                 if tmpMin < MIN:
                     MIN = tmpMin
             MAX = MAX*1.0
             MIN = MIN*1.0
    else:
       assert options.determine is None

    minMaxDiff = MAX - MIN
    print 'MAX = ', MAX
    print 'MIN = ', MIN
    assert minMaxDiff > 0
    
    if options.determine:
      print 'no file written.'
      sys.exit(0)

    nslices = vigra.impex.numberImages(in_tiffs[0])
    if nslices == 1:
       prefix = 'txy_'
    else:
       prefix = 'txyz_'

    out_fn = prefix + os.path.splitext(os.path.basename(in_tiffs[0]))[0] + "-" + os.path.splitext(os.path.basename(in_tiffs[-1]))[0] + ".h5"
    print out_fn
    f = h5py.File(out_fn,'w')

    for idx,t in enumerate(in_tiffs):
        if idx > options.end and options.end != -1:
            break
        #img = vigra.readImage( t , dtype=dataType, index=0 )
        nslices = vigra.impex.numberImages(t)
        img = [vigra.readImage( t , dtype=dataType, index=i ) for i in range(nslices)]
        img = np.dstack(img)
        img = np.array(img)
        # v = np.array(img)
        if npType == np.uint16:
           v = np.array((img-MIN)/minMaxDiff*255).astype(np.uint8)
        else:
           v = np.array(img).astype(np.uint8)

        a = v
        if idx == 0:           
           if options.end > -1:
              shape = [options.end]
           else:
              shape = [len(in_tiffs)]
           shape.extend(a.shape)
           chunks = [1,]
           if len(a.shape) == 3:
               axistags = 'txyc'
           elif len(a.shape) == 4:               
               axistags = 'txyzc'
           else:
               raise Exception("shape not supported: " + str(shape))
           for s in a.shape:
               if s > 64:
                  chunks.append(64)
               else:
                  chunks.append(s)
           try:
              ds = f.create_dataset('/volume/data', shape, compression=1, dtype=a.dtype, chunks=tuple(chunks))
           except:
              print 'shape, a.dtype, chunks ', (shape, a.dtype, chunks)
              raise
	
        ds[idx,...] = a

    ds.attrs['axistags'] = vigra.defaultAxistags(axistags).toJSON()
    f.close()


