# C++ Pipeline to run Conservation Tracking

by Philipp Hanslovsky, Carsten Haubold, Martin Schiegg, David Stoeckel

University of Heidelberg, 2015

## About

This pipeline runs the conservation tracking algorithm given a sequence of tiff images and pre-trained Random Forest classifiers.
The classifiers can be extracted from previously trained [ilastik](http://ilastik.org) projects for pixel classification and tracking,
but every step will be rerun in the C++ pipeline (allows for batch-segmenting and tracking).

It was originally developed for the ISBI 2015 Cell Tracking Challenge, and is thus only operating on tiff sequences.

## Compilation

*Disclaimer*: this was never compiled or tested on Windows!

The easiest way to get this compiled is by having a [conda setup](https://github.com/ilastik/ilastik-build-conda) of `ilastik-everything`, let's assume that the respective conda environment is called `ilastikdev`. You also need the https://github.com/ilastik/ilastik-build-conda repository cloned somewhere locally.

Then you need to clone [pgmlink](https://github.com/martinsch/pgmlink) yourself to check out a different (isbi_challenge_15) branch as follows
```
# in your favorite location:
git clone http://github.com/martinsch/pgmlink
cd pgmlink

# Just run the build script with the right environment setup
PREFIX=/path/to/your/miniconda/envs/ilastikdev \
PYTHON=/path/to/your/miniconda/envs/ilastikdev/bin/python2.7 \
CPU_COUNT=4 \
bash /path/to/your/ilastik-build-conda/pgmlink/build.sh

# now check out the proper pgmlink branch
git checkout isbi_challenge_15
cd build
make install
```

With this pgmlink version installed, you can clone, configure and build this C++ pipeline:
```
git clone https://github.com/chaubold/conservation_tracking_pipeline
cd conservation_tracking_pipeline
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX /path/to/your/miniconda/envs/ilastikdev -DCPLEX_ROOT_DIR /path/to/your/cplex/root
make -j
```

## Usage

Let's assume you trained a pixel classification project `pc.ilp`, 
exported the prediction maps, and then set up an automated tracking workflow `tracking.ilp`. 
We suggest to run tracking for only a small field of view to see that the chosen parameters yield sensible results.
Save after running tracking to make sure the parameters are up to date in the file `tracking.ilp`

To create the configuration files for this C++ pipeline, use the following command:
```
python /path/to/this/repo/scripts/ilpToPipelineConfig.py --pixel-classification-project /path/to/pc.ilp --tracking-project /path/to/tracking.ilp --out /my/existing/out/dir
```

This will create a set of files in `/my/existing/out/dir`:
* `classifier.h5`: the extracted random forests for pixel classification, object count detection and division detection
* `features.txt`: the selected pixel classification features
* `object_count_features.txt`: the selected features for object count classification
* `division_features.txt`: the selected features for division classification
* `tracking_config.txt`: a text file containing all other specified parameters

Using these files, the raw tiff images in `/path/to/dataset/raw_images` and two empty existing directories for the resulting segmentation (after pixel classification and thresholding) `/path/to/dataset/segmentations`, as well as resulting relabeled segmentations after tracking (where each tracked object is colored in its track ID) `/path/to/dataset/result`, the pipeline can be run as follows:

```
python /path/to/this/repo/build/isbi_pipeline /path/to/dataset/raw_images /path/to/dataset/segmentations /path/to/dataset/result /my/existing/out/dir/tracking_config.txt /my/existing/out/dir/classifier.h5 /my/existing/out/dir/features.txt /my/existing/out/dir/object_count_features.txt /my/existing/out/dir/division_features.txt
```

The file naming conventions for input and resulting files are subject to the Cell Tracking Challenge [Guidelines](http://ctc2015.gryf.fi.muni.cz/Public/Documents/Naming%20and%20file%20content%20conventions.pdf)

#### Notes:
* The `isbi_pipeline32` and `tracking32` executables are using a label space of type `uint32` whereas the default pipeline uses `uint16`. This is because one dataset in the ISBI challenge exceeded the number of labels representable by 16 bit.
* The `tracking` executables are meant to work on previously obtained segmentations. Thus after running the `isbi_pipeline` once, you only need to run `tracking` again if you did not change of the parameters related to pixel classification.

## References

* M. Schiegg, P. Hanslovsky, B. X. Kausler, L. Hufnagel, F. A. Hamprecht. **Conservation Tracking**. Proceedings of the IEEE International Conference on Computer Vision (ICCV 2013), 2013.
* [Cell Tracking Challenge](http://www.codesolorzano.com/celltrackingchallenge/Cell_Tracking_Challenge/)