# Procedural and Expandable Neural Terrain Generation with Transfomers
## A Pytorch implementation of a maskGIT-based pipeline to generate infinitely expandable procedural terrain based on real-world geographical data

Existing approaches to procedural generation of terrain fail to account for the diversity of geology and topography evident in the real world, or fail to do it scalably. 
- Noise-based approaches require tuning careful tuning of parameters to produce convincing results since they have no physical basis. Furthermore, they suffer from artifacting and repetitiveness.
- Erosion-based approaches produce great quality for one region of terrain, but the produced terrain cannot be expanded easily or and the process cannot be done online. These approaches also do not scale well with increased map size
- GANs also do not scale well with increased map size. Also, they have no sense of long-range relations, which are important in real terrain (ie. rivers ending in the sea or mountains sloping off into plains).

By leveraging a transformer trained on the abstract space of a VQGAN, this model aims to produce topologically and geologically coherent height maps with meaningful long-range relations and infinite extensibility. 

Objectives:
1) Implement maskGIT and generate height maps based on real world elevation data.
2) Integrate climatological data and geological data to provide coherent climate and rock types which can inform later steps of generation (vegetation, surface material, weather) and produce meaningful environements
3) (Optional) Replace maskGIT scheduler to allow for explicit control over the order in which tokens are generated

## Objective 1:

## Stage 1 [Completed]
### Implement VQVAE

I started with verifying that I could replicate results from VQVAE on a similar dataset. 
Below are results from VQVAE after training on STL10 for 50 epochs (Latent resolution of 8x8x256, codebook size 1024).
![alt text](vqvae_example.png)
Reconstruction quality is passable, expecially considering the low latent resolution. There is clear blockiness in areas that should be soft gradients (clouds, sky), and loss of high frequency information (boat windows).

## Stage 2 [Completed]
### Implement discriminator

The VQGAN is trained without the discriminator loss applied to the VAE for some initial period. Once the loss started to be incorporated into the VAE loss function, however, the discriminator quickly lost its ability to discriminate and develoved into noise. The VQGAN's adaptive loss scaling, which increases the weight of the discriminator loss relative to the combined perceptual and reconstruction loss amplified this noise since the gradients in the discriminator were still very large. As such, the loss function was dominated by the spurious discriminator signal to the detriment of the reconstruction quality. To improve the quality of the signal from the discriminator, I applied a scaling factor to the generator loss inversely proportional to the discriminator loss. Thus, when the discriminator loss is very high, the generator largely trains only on the perceptual / reconstruction loss. As the discriminator gains power, the generator pays more attention to its feedback. As such, the generator and discriminator are more balanced, with neither dying out due to the other getting too good. This, as well as some other small changes such as the adoption of hinge loss for the discriminator and non-saturating loss for the generator were able to stabilize training and produce good results:

STL10 (50 epochs):
![alt text](image-1.png)
Note the superior color reproduction on the birds and sharper detail in the sky. While the boat and the far right image show better sharpness, they also have some clear artifacting.

USGS DEM (100 epochs, tiff converted to png for memory conservation):
![alt text](image-2.png)
The reconstructions successfully captures the general contours of the terrain, but struggle somewhat still with high frequency information and oversmooths some gradients.

## Stage 3 [WIP]
### Implement transformer and mask scheduler

While the reconstruction quality can still be refined, it appears to be good enough to start implementing and training the transformer. I am in progress implementing the BERT-based bidirectional transformer from MaskGIT.

## Objective 2:



## Objective 3:

The maskGIT scheduler simultaneously predicts all tokens, discards the least confident, and repeats. If the terrain generation is performed only once, this is fine, but in order to continually outpaint the terrain while appropriately considering the already existing terrain, it would be preferable to control which tokens are generated to reduce sequence length and computational overhead.

The scheduler will include only already-generated patches and select patches on the fringe, discarding unneeded masked tokens at the fringe. Thus, we can maximize context while minimizing sequence length, while still retaining the option to expand the terrain more at a later iteration.

## Further extensions:

Limitations in processing power makes high-resolution synthesis with transformers alone impractical on consumer hardware. Diffusion-based super resolution could allow for post hoc augmentation of generated height map resolution without costly erosion simulation or high-resolution patches.

## Usage:

1) Make virtual environment with Python >= 3.12. Install requirements.txt.
2) Download STM10 datset or DEM tiff from links below. If using tiff data, use _notebooks/slice_geo_data.ipynb to process the tiff into smaller png patches and sample a training dataset.
3) Train with ```python3 train_vqgan.py --config {config name}```, where config name is the name of a config file in ./configs. Training using the flag ```--checkpoint ```, which will use the latest checkpoint unless otherwise specified. Output directory can be specified with ```--save_as name```.
4) Visualize reconstruction quality with ```python3 evaluate.py --checkpoint_dir directory --checkpoint checkpoint.pt```
5) Training statistics are visualized in _notebooks/training_stats.ipynb

## Datasets:

https://data.usgs.gov/datacatalog/data/USGS:77ae0551-c61e-4979-aedd-d797abdcde0e
https://www.cec.org/files/atlas/?z=4&x=-93.3838&y=43.1651&lang=en&layers=climatezones&opacities=100&labels=true
https://cs.stanford.edu/~acoates/stl10/

# References:



## Hardware:

All training and evalution performed on an NVIDIA RTX 5070ti with 16gb of VRAM.