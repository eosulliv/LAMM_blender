LAMM Blender Add-On
===================

This is the official Blender add-on repository for the paper [Locally Adaptive Neural 3D Morphable Models](https://arxiv.org/pdf/2401.02937.pdf).

This repository provides the Blender add-on that faciliates manual interaction and manipulation of the mesh shape via control points.

The latest version of this add-on, ``lamm_blender_x_xx.zip'', can be downloaded from the release section of this repository.

To install the add on:
----------------------
**Method 1:**
- Unzip the file "lamm_blender_x_xx.zip" into the "<>\scripts\addons" Blender directory
# To do  - Make this clearer

or

**Method 2:**
- Open Blender
- Navigate to Edit > Preferences > Add-ons
- Click on the "Install" button
- Navigate to the downloaded directory in your file directory
- Double-click the zip file to install it

Downloading the model and checkpoints
-------------------------------------
**Downloading LAMM:**
Navigate to the "<>\scripts\addons\lamm_blender_x_xx" directory via the command line.
Clone [LAMM](https://github.com/michaeltrs/LAMM) from the github repository GitHub repository.

**Model checkpoints:**
The add-on assumes the checkpoint is located within "LAMM/assets/checkpoints."
You can change this and point the add-on to the checkpoint folder of your trained model using the file search option at the top of the "Face Model" pane.

The checkpoint directory is expected to contain the following assets:
- checkpoint.pth
- config_file.yaml
- files
  - displacement_stats.pickle
  - gaussian_id.pickle
  - mean_std.pickle
  - region_boundaries.pickle
  - region_ids.pickle
  - region_names.pickle (optional)
  - template.obj


Set-up
------
Once the plug-in has been installed and the model downloaded, you will need to installed the python libraries required to let it run. A script named ``setup.py'' has been provided for this purpose.

- Open setup.py in the Blender scripting pane
- Run the script and wait for all packages to install
- You may need to restart Blender after all packages have completed installation


Acknowledgements
----------------
If you found this work interesting and would like to use it in future projects, please cite:
(insert bibtex for the paper here)

