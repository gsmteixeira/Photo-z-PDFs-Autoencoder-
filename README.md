# Photo-z-PDFs-Autoencoder-
Example code of how to train and validate the autoencoder for photometric redshifts PDFs from MDN

## This repo contains:
Training set DLTRAIN-A.

Reduced test set (clpoud storage reasons).

PDF parameters generated from the MDN for the entire DLTRAIN-A.

PDF parameters generated from the MDN for the reduced test set.

PDFs generated for the test set defined in the range (0<z<2).

AE archtecture and hyperparameters (AUTOENCODER_training.py).

Notebook for analysis (analysis.ipynb).

## Steps:
Run AUTOENCODER_training.py to train the AE model.

Navigate through the analysis.ipynb to compare the decoded results with the results from the MDN model, and the spec-z. 

## Reference
https://arxiv.org/abs/2408.15243v1
