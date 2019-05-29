
# coding: utf-8

# <h1>Audio Spectrum Approximation using Over-Complete 2D Patch Dictionaries</h1>
# <h2>Michael A. Casey, 2014-2015 Bregman Media Labs, Dartmouth College</h2>

# In[1]:

import sys

from pylab import *
from bregman.suite import *
import AudioSpectrumPatchApproximation as A # 2D spectrogram patch sparse approximation library
#reload(A) # for debugging
# get_ipython().magic('matplotlib inline')
# get_ipython().magic('matplotlib inline')
rcParams['figure.figsize'] = (12.0, 8.0) # Make larger in-line figures


# <h2>Constant-Q Time-Frequency Analysis</h2>

# In[2]:


#Analyze 10s of a natural scene (field recording) using a log frequency scale (constant-Q transform) 
# F = LogFrequencySpectrum('chernobyl.wav',nhop=1024, nfft=8192, wfft=4096, npo=24) # constant-Q transform
F = LogFrequencySpectrum('forest1.wav',nhop=1024, nfft=8192, wfft=4096, npo=24) # constant-Q transform

#Plot the spectrum series and the time-averaged constant-Q spectrum.
pargs = {'normalize':True, 'dbscale':True, 'cmap':cm.hot, 'vmax':0, 'vmin':-45} # plot arguments
subplot(211); F.feature_plot(nofig=True, **pargs) # plot the transform
title('Constant-Q Spectrum (Log Amplitude)', fontsize=14)
subplot(212); plot(20*log10(1+F.X).sum(1)) # and the time-averaged constant-Q spectrum
title('Mean Constant-Q Spectrum: Log Amplitude ', fontsize=14)
xticks(arange(0,100,8), F._logfrqs[0:-1:8].round())
xlabel('Frequency (Hz)'); ylabel('Log Amplitude')
grid();axis('tight');ax=colorbar();ax.ax.set_visible(False)

# <h3>Inverting the analysis back to audio (baseline resynthesis)</h3>

# In[71]:


#Invert the constant-Q transform to an audio signal, using inverse constant-Q transform
## xh = F.inverse()
## play(balance_signal(xh)) # Check the resynthesis without sparse coding


# <h2>Method 1. Sparse approximation dictionary learning, using mini bach gradient learning</h2>

# In[41]:


# Learn sparse codes from data using dictionary learnng on n x m patches of the constant-Q transform.
s1 = A.SparseApproxSpectrum(patch_size=(8,8)) # Set spectrum 2D patch size here (n,m) 
s1.extract_codes(F.X, n_components=9, alpha=1, zscore=True, log_amplitude=True) # Learn a dictionary of patches for spectrogram
s1.plot_codes(cbar=True, cmap=cm.hot) # show the learned codes

# In[42]:


# Reconstruct the constant-Q spectrogram using each learned patch basis
s1.reconstruct_individual_spectra(plotting=True, **pargs)
figure()
subplot(211); feature_plot(F.X, nofig=True, **pargs); title('Original Spectrogram', fontsize=14)
subplot(212); feature_plot(s1.X_hat, nofig=True, **pargs); title('Sparse Approximation Spectrogram', fontsize=14)


# In[37]:


X_hat = s1.X_hat_l[2] # <- change the patch to reconstruct here
x_hat = F.inverse(X_hat) # , Phi_hat=(rand(*F.STFT.shape)*2-1)*pi) 
play(balance_signal(x_hat))
feature_plot(X_hat, **pargs); title('1 Component Reconstruction', fontsize=14)


# <h2>Method 2: Decomposition with a "Gabor field" dictionary</h2>

# In[75]:

sys.exit()



# Apply 2D Gabor field to n x m patches of the constant-Q transform.
s2 = A.SparseApproxSpectrum(patch_size=(12,12))
s2.make_gabor_field(F.X, thetas=arange(0,4), sigmas=(2,3.5), frequencies=(0.05,0.15), zscore=True, log_amplitude=True) # Generate a dictionary of Gabor patches for spectrogram
s2.plot_codes(cbar=True, cmap=cm.hot) # show the learned codes


# In[76]:


# Reconstruct the constant-Q spectrogram using each learned patch basis
s2.reconstruct_individual_spectra(plotting=True, **pargs)
figure()
subplot(211); feature_plot(F.X, nofig=True, **pargs); title('Original Spectrogram', fontsize=14)
subplot(212); feature_plot(s2.X_hat, nofig=True, **pargs); title('Sparse Approximation', fontsize=14)


# In[77]:


X_hat = s2.X_hat_l[13] # <- change the patch to reconstruct here
x_hat = F.inverse(X_hat) #, Phi_hat=(rand(*F.STFT.shape)*2-1)*pi) 
play(balance_signal(x_hat))
feature_plot(X_hat, **pargs); title('1 Component Reconstruction', fontsize=14)


# <h2>Method 3: Decomposition with SIPLCA2D component dictionary<h2>

# In[91]:


# Learn Non-Negative Components of n x m patches of the constant-Q transform.
s3 = A.SparseApproxSpectrumPLCA2D(patch_size=(12,8)) # Same idea as above, but with non-negative components
""" Hyper-parameters:
alphaW, alphaZ, alphaH : float or appropriately shaped array
    Dirichlet prior parameters for `W`, `Z`, and `H`.
    Negative values lead to sparser distributions, positive
    values makes the distributions more uniform.  Defaults to
    0 (no prior).

    **Note** that the prior is not parametrized in the
    standard way where the uninformative prior has alpha=1.
betaW, betaZ, betaH : non-negative float
    Entropic prior parameters for `W`, `Z`, and `H`.  Large
    values lead to sparser distributions.  Defaults to 0 (no
    prior).
nu : float
    Approximation parameter for the Entropic prior.  It's
    probably safe to leave the default.
"""
s3.extract_codes(F, n_components=6, log_amplitude=True, alphaW=0.0, alphaZ=0.0, alphaH=0.0, betaW=0.00, betaZ=0.001, betaH=0.00)
s3.plot_codes(cbar=True, cmap=cm.hot)


# In[92]:


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


# <h1>Help on AudioSpectrumPatchApproximation</h1>

# In[99]:


help(A)

