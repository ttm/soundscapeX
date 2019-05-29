# experiments for research in ecological soundscapes

## exp1

Following the results reported in this article:
  https://peerj.com/articles/2108/

Had to use Python 2 to avoid complicated translation of numeric types in porting the needed librartis.

Had to do some simple conversion by hand (numpy is not accepting float as array index anymore) to install:
  https://github.com/bregmanstudio/BregmanToolkit

And change
  from sklearn.decomposition import ProjectedGradientNMF
to 
  from sklearn.decomposition import NMF as ProjectedGradientNMF
in order to use:
  https://github.com/bregmanstudio/AudioSpectrumPatchApproximation/

Then the scripts in exp1/ folder execute alright.
We should check with the authors if NMF is giving the expected results.
It seems to be correct, but these classes have changed a lot.

:::

