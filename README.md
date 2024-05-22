Daniel Egaña-Ugrinovic, 2024.

This is a python/pytorch code to perform image reconstruction of distant black holes taking as input simulated intensity-interferometry measurements (magnitude of the image's Fourier transform).

It can reconstruct the black hole parameters using Decision Trees, Random Forests or CNNs. It can also perform full parameter-agnostic image reconstruction using CNNs, starting from just the magnitude of the Fourier Transform.

CNNAGNvx has the main code. It calls main.c, a third-party AGN code (credit: Neal Dalal) to generate the AGN images, 
that I modified for python and cluster integration.

