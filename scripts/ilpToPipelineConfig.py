import vigra
import numpy as np
import h5py
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Create a config.txt, classifier.h5 and features.txt from a given pixel classification and tracking ilastik project")
	#parser.add_argument('--shape', metavar='shape', required=True, type=int, nargs='+', help='Shape of each image given as 2 or 3 ints (time axes is handled internally)')
	parser.add_argument('--pixel-classification-project', dest='pc_project', required=True, type=str, help='Filename of ilastik pixel classification project')
	parser.add_argument('--tracking-project', dest='tracking_project', required=True, type=str, help='Filename of ilastik tracking project')
	parser.add_argument('--out', dest='out_path', default='.', type=str, help='Output folder, default CWD')
	args = parser.parse_args()

	with h5py.File(args.pc_project, 'r+') as in_f:	
		# ------------------------------------------------------
		# copy pixel classification feature selection
		scales = in_f['FeatureSelections/Scales']
		featureNames = in_f['FeatureSelections/FeatureIds']
		selectionMatrix = in_f['FeatureSelections/SelectionMatrix'].value

		assert(selectionMatrix.shape[0] == featureNames.shape[0])
		assert(selectionMatrix.shape[1] == scales.shape[0])

		with open(args.out_path + '/features.txt', 'w') as out_f:
			for f in range(featureNames.shape[0]):
				for s in range(scales.shape[0]):
					if selectionMatrix[f][s] == True:
						out_f.write('{},{}\n'.format(featureNames[f], scales[s]))

		# ------------------------------------------------------
		# copy pixel classification random forest
		with h5py.File(args.out_path + '/classifier.h5', 'w') as out_f:
			out_rf = out_f.create_group('PixelClassification')
			in_f.copy('PixelClassification/ClassifierForests', out_rf)

		# ------------------------------------------------------
		# copy tracking configuration
		# TODO...