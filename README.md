# EnteroidSeg

EnteroidSeg is an example of image processing pipeline developed to identify nuclei and cell-types in the 2d enteroid cultures [\[1\]](#ref1). Example code for segmentation of nuclei, EdU+ nuclei, stem cells, and goblet cells are provided. This set of code accompanies a paper on 2d enteroid culture and analysis, which should be cited for this code [\[2\]](#ref2).

## Setup

The package was developed with Python. 

### Method 1
The easiest way to set up the appropriate environment is through conda. Miniconda3 can be install ed https://docs.conda.io/en/latest/miniconda.html

After installing Miniconda, download the package and navigate to the **EnteroidSeg** folder. Setup the Python environment with the following command.

```
conda env create --file=environment.yaml
```

A conda environment named `enteroidseg` will be created. Activate the environment to run the scripts in this package. The environment can be activated with the following command

```
source activate enteroidseg
```

### Method 2
The required Python version and packages can also be installed manually. The required version and packages are listed in [requirements.txt](requirements.txt)

## Usage

Segmentation using the provided sample images (under **images**) can be done by running scripts for each segmentation type. 

### Nuclear Segmentation

To run the nuclear segmentation, navigate to the **enteroidseg** folder and execute the following to command.

```
python nuclear_segmentation.py
```

The output the of the script consists of three files and will be store in **output**
- dna_segmentation.tiff: segmentation file with labeled segmented objects
- dna_overlay.png: visualization of segmentation result overlayed on the raw input image
- dna_pipeline_result.png: visualization of main steps in the pipeline

### EdU Segmentation

To run the EdU+ nuclei segmentation, navigate to the **enteroidseg** folder and execute the following to command.

```
python edu_segmentation.py
```

The output the of the script consists of three files and will be store in **output**
- edu_segmentation.tiff: segmentation file with labeled segmented objects
- edu_overlay.png: visualization of segmentation result overlayed on the raw input image
- edu_pipeline_result.png: visualization of main steps in the pipeline

### Stem Segmentation

To run the stem cell segmentation, navigate to the **enteroidseg** folder and execute the following to command.

```
python stem_segmentation.py
```

The output the of the script consists of three files and will be store in **output**
- stem_segmentation.tiff: segmentation file with labeled segmented objects
- stem_overlay.png: visualization of segmentation result overlayed on the raw input image
- stem_pipeline_result.png: visualization of main steps in the pipeline

### Goblet Segmentation

To run the goblet cell segmentation, navigate to the **enteroidseg** folder and execute the following to command.

```
python goblet_segmentation.py
```

The output the of the script consists of three files and will be store in **output**
- goblet_segmentation.tiff: segmentation file with labeled segmented objects
- goblet_overlay.png: visualization of segmentation result overlayed on the raw input image
- goblet_pipeline_result.png: visualization of main steps in the pipeline

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## References

<a name="ref1">1</a>: Thorne, C.A.\*, Chen, I.W.\*, Sanman, L.E., Cobb, M.H., Wu, L.F., and Altschuler, S.J. (2018). Enteroid Monolayers Reveal an Autonomous WNT and BMP Circuit Controlling Intestinal Epithelial Growth and Organization. Dev. Cell 44, 624–633.e4.
<a name="ref2">2</a>: (TODO: cite methods paper)