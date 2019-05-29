
from pylab import *
from bregman.suite import *
import AudioSpectrumPatchApproximation as A # 2D spectrogram patch sparse approximation library
F = LogFrequencySpectrum('forest1.wav',nhop=1024, nfft=8192, wfft=4096, npo=24) # constant-Q transform
pargs = {'normalize':True, 'dbscale':True, 'cmap':cm.hot, 'vmax':0, 'vmin':-45} # plot arguments

s3 = A.SparseApproxSpectrumPLCA2D(patch_size=(12,8)) # Same idea as above, but with non-negative components
s3.extract_codes(F, n_components=3, log_amplitude=True, alphaW=0.0, alphaZ=0.0, alphaH=0.0, betaW=0.00, betaZ=0.001, betaH=0.00)
s3.plot_codes(cbar=True, cmap=cm.hot)


s3.reconstruct_individual_spectra()
s3.plot_individual_spectra(**pargs)
figure()
subplot(211); feature_plot(F.X, nofig=True, **pargs); title('Original Spectrogram', fontsize=14)
subplot(212); feature_plot(s3.X_hat, nofig=True, **pargs); title('Sparse Approximation', fontsize=14)


# In[98]:


X_hat = s3.X_hat_l[0] # <- change the patch to reconstruct here
x_hat = F.inverse(X_hat) #, Phi_hat=(rand(*F.STFT.shape)*2-1)*pi) 
play(balance_signal(x_hat))
feature_plot(X_hat, **pargs); title('1 Component Reconstruction', fontsize=14)
