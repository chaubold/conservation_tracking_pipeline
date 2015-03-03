import vigra
import numpy as np
import h5py
import argparse

def writeSelectedRegionFeatures(out_f, h5_group):
	for feature_group_name in h5_group.keys():
		feature_group = h5_group[feature_group_name]
		for feature in feature_group.keys():
			# discard squared distances feature
			if feature == 'ChildrenRatio_SquaredDistances':
				continue

			if feature == 'Coord<Principal<Kurtosis>>':
				feature = 'Coord<Principal<Kurtosis> >'
			elif feature == 'Coord<Principal<Skewness>>':
				feature = 'Coord<Principal<Skewness> >'
			out_f.write('{}\n'.format(feature))

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

		with h5py.File(args.out_path + '/classifier.h5', 'w') as out_f:
			# ------------------------------------------------------
			# copy pixel classification random forest
			out_rf = out_f.create_group('PixelClassification')
			in_f.copy('PixelClassification/ClassifierForests', out_rf)

	with h5py.File(args.tracking_project, 'r+') as in_f:	
		with h5py.File(args.out_path + '/classifier.h5', 'r+') as out_f:
			# ------------------------------------------------------
			# copy tracking random forests
			out_rf = out_f.create_group('CountClassification')
			in_f.copy('CountClassification/ClassifierForests', out_rf)

			out_rf = out_f.create_group('DivisionDetection')
			try:
				in_f.copy('DivisionDetection/ClassifierForests', out_rf)
			except:
				if in_f['ConservationTracking/Parameters/0/withDivisions'].value:
					raise Exception("Did not find DivisionDetection/ClassifierForests even though withDivisions is selected")
				in_f.copy('CountClassification/ClassifierForests', out_rf)
				print("Copying dummy classifier for divisions")

			# ------------------------------------------------------
			# copy tracking configuration
			#out_tracking = out_f.create_group('ConservationTracking')
			in_f.copy('ConservationTracking', out_f)

		with open(args.out_path + '/object_count_features.txt', 'w') as out_f:
			writeSelectedRegionFeatures(out_f, in_f['CountClassification/SelectedFeatures'])

		with open(args.out_path + '/division_features.txt', 'w') as out_f:
			writeSelectedRegionFeatures(out_f, in_f['DivisionDetection/SelectedFeatures'])

		threshold_level = in_f['ThresholdTwoLevels/SingleThreshold'].value
		threshold_channel = in_f['ThresholdTwoLevels/Channel'].value

		with open(args.out_path + '/tracking_config.txt', 'w') as out_f:
			params = in_f['ConservationTracking/Parameters/0']
			out_f.write('tracker,ConsTracking\n')
			out_f.write('borderWidth,0\n')
			out_f.write('forbiddenCost,0\n')
			out_f.write('epGap,0.05\n')
			out_f.write('templateSize,50\n')
			out_f.write('transParameter,5\n')
			out_f.write('withConstraints,1\n')
			out_f.write('detWeight,10.0\n')
			out_f.write('Channel,{}\n'.format(threshold_channel))
			out_f.write('SingleThreshold,{}\n'.format(threshold_level))
			if params['z_range'][1] - params['z_range'][0] == 1:
				out_f.write('nDim,2\n')
			else:
				out_f.write('nDim,3\n')

			for key in params.keys():
				try:
					if len(params[key].value) > 1:
						for i in range(len(params[key].value)):
							out_f.write('{}_{},{}\n'.format(key, i, params[key].value[i]))
					elif len(params[key].value) == 1:
						out_f.write('{},{}\n'.format(key, params[key].value[0]))
					else:
						raise Exception("nothing")
				except:
					if params[key].dtype == np.dtype('bool'):
						if params[key].value:
							out_f.write('{},1\n'.format(key))
						else:
							out_f.write('{},0\n'.format(key))
					else:
						if key == 'cplex_timeout' and params[key].value == '':
							out_f.write('{},{}\n'.format(key, 1e+75))
						else:
							out_f.write('{},{}\n'.format(key, params[key].value))
