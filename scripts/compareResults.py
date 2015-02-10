import vigra
import numpy as np

cd /Users/chaubold/hci/projects/cell_tracking_challenge_15/Fluo-N2DH-SIM/01_RES
ls
ls | grep tif
files = !ls | grep tif

for f in files:
	a = vigra.impex.readImage('/Users/chaubold/hci/projects/cell_tracking_challenge_15/Fluo-N2DH-SIM/01_RES/' + f)
	b = vigra.impex.readImage('/Users/chaubold/hci/projects/cell_tracking_challenge_15/Fluo-N2DH-SIM/01_RES_ORIG/' + f)
	print (a == b).all()