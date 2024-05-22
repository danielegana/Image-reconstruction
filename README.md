Daniel Egaña-Ugrinovic, 2024.

This is a python/pytorch code to perform image reconstruction of distant black holes taking as input simulated intensity-interferometry measurements (magnitude of the image's Fourier transform).

It can reconstruct the black hole parameters using Decision Trees, Random Forests or CNNs. It can also perform full parameter-agnostic image reconstruction using CNNs, starting from just the magnitude of the Fourier Transform.

CNNAGNvx has the main code. CNNAGN requires input images for training. Here we generate the input images calling a third-party code main.c (credit: Neal Dalal), that I modified for python and cluster integration.

