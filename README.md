# IMGA_cuda
An open source GPU accelerated implementation of the Island Model Genetic Algorithm. You can experiment with different population sizes, parent selection functions, survivor selection functions and other hyper parameters in the files that end with _island.py. To run the _island.py file, you need to specify two command line arguments: devide and num_islands. Device could be cpu or gpu and num_islands could be any number. 

The below 3 files are together a single problem's implementation on the Island model.
taichi_{problemname}_readdata.py
taichi_{problemname}.py
taichi_{problemname}_island.py

(Graph coloring algoritm does not work properly i.e. genome does not represent its fitness score)

taichi_rng.py files is for our unique random number generator.

Corresponding CPU implementation of Island models is in its island_model_nonGPU folder but CPU implementation of the normal GA is present in any '{problemname}.py' file.

## Colab Notebook
The given notebook gives step by step instructions to run the code on Google Colab using (GPU). Please look at the notebook to understand the implementation and the results.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17N6HOzigsnEE9V8ozaqKU87fzNkzwpGB?usp=sharing) 