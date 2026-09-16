# Procedural and Expandable Neural Terrain Generation with Transformers

A PyTorch implementation of a MaskGIT transformer pipeline for autoregressive terrain outpainting using real-world USGS topographical elevation data.

📄 **[Read the Full Project Writeup & Qualitative Results (Google Doc)](https://docs.google.com/document/d/1PaDbwQHK_b1MDOYDmEfIzp2OyKiYvBFKlIT0UFi4dOo/edit?usp=sharing)**

## Abstract:

Existing approaches to procedural generation of terrain fail to account for the diversity of geology and topography evident in the real world, or fail to do it extensibly.
- Noise-based approaches have no physical basis and require careful tuning of parameters to produce convincing results. Furthermore, they suffer from artifacting and repetitiveness.
- Erosion-based approaches produce great quality for one region of terrain, but the tproduced terrain cannot be expanded easily and the process usually cannot be done online.
- GANs also do not scale well with increased map size. Also, they have no sense of long-range relations, which are important in real terrain (ie. rivers ending in the sea or mountains sloping off into plains).

By leveraging a transformer trained on the abstract space of a VQGAN, this model aims to produce topologically and geologically coherent height maps with meaningful long-range relations and infinite extensibility.
Currently, the model is trained on a small dataset and is capable of outpainting a seed tile of terrain data in one dimension. Immediate next steps include retraining on a larger dataset at a higher height resolution and implementing a conditional decoder to enable iterative expansion and decoding of the terrain latent space.


## Project Overview
- **Architecture:** Quantized VQGAN latent space paired with a MaskGIT transformer model.
- **Dataset:** USGS National Map 3D Elevation Program (3DEP) 1m resolution GeoTIFF data.
- **Goal:** Autoregressively generate geologically coherent, infinitely expandable terrain height maps.

## Usage:

1) Make virtual environment with Python >= 3.12. Install requirements.txt.
2) Download STM10 datset or DEM tiff from links below. If using tiff data, use ```_notebooks/slice_geo_data.ipynb``` to process the tiff into smaller png patches and sample a training dataset.
3) Train with ```python3 train_vqgan.py --config {config name}```, where config name is the name of a config file in ./configs. Training using the flag ```--checkpoint ```, which will use the latest checkpoint unless otherwise specified. Output directory can be specified with ```--save_as name```.
4) Visualize reconstruction quality with ```python3 evaluate.py --checkpoint_dir directory --checkpoint checkpoint.pt```
5) Losses and other performance metrics are logged automatically and can be visualized with ```_notebooks/training_stats.ipynb```

## Datasets:

- https://cs.stanford.edu/~acoates/stl10/
- https://data.usgs.gov/datacatalog/data/USGS:77ae0551-c61e-4979-aedd-d797abdcde0e
- https://www.cec.org/files/atlas/?z=4&x=-93.3838&y=43.1651&lang=en&layers=climatezones&opacities=100&labels=true

# References:

[1]
H. Chang, H. Zhang, L. Jiang, C. Liu, and W. T. Freeman, ‘MaskGIT: Masked Generative Image Transformer’, arXiv [cs.CV]. 2022.

[2]
P. Esser, R. Rombach, and B. Ommer, ‘Taming Transformers for High-Resolution Image Synthesis’, arXiv [cs.CV]. 2021.

[3]
A. van den Oord, O. Vinyals, and K. Kavukcuoglu, ‘Neural Discrete Representation Learning’, arXiv [cs.LG]. 2018.

## Hardware:

All training and evaluation performed on an NVIDIA RTX 5070ti with 16gb of VRAM.
