
from pylab import *
from bregman.suite import *
import numpy as n
import AudioSpectrumPatchApproximation as A
from sklearn.decomposition import NMF as ProjectedGradientNMF

# step 1: obtain frequency patterns from a short segment
F_ = LogFrequencySpectrum('birds_.wav',nhop=1024, nfft=8192, wfft=4096, npo=24)

s3 = A.SparseApproxSpectrumPLCA2D(patch_size=(12,8))
s3.extract_codes(F_, n_components=3, log_amplitude=True, alphaW=0.0, alphaZ=0.0, alphaH=0.0, betaW=0.00, betaZ=0.001, betaH=0.00)


###############
# step 2: find the components of the complete audio using the frequency patterns found:
F = LogFrequencySpectrum('forest1.wav',nhop=1024, nfft=8192, wfft=4096, npo=24)

# F.h is missing...
## SIPLCA2.reconstruct(F_.w, F_.z, F.h)
