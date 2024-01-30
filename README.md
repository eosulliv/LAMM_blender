LAMM Blender Add-On
===================

This is the official Blender add-on repository for the paper [Locally Adaptive Neural 3D Morphable Models](https://arxiv.org/pdf/2401.02937.pdf).

This repository provides the Blender add-on that faciliates manual interaction and manipulation of the mesh shape via control points.

The latest version of this add-on, ``lamm_blender_x_xx.zip'', can be downloaded from the release section of this repository.

To install the add on:
----------------------
Method 1:
- Unzip the file "lamm_blender_x_xx.zip" into the "<>\scripts\addons" Blender directory

or

Method 2:
- Open Blender
- Navigate to Edit > Preferences > Add-ons
- Click on the "Install" button
- Navigate to the downloaded directory in your file directory
- Double-click the zip file to install it

Downloading the model and checkpoints
-------------------------------------
Downloading LAMM:
- Navigate to the "<>\scripts\addons\lamm_blender_x_xx" directory via the command line.
- Clone [LAMM](https://github.com/michaeltrs/LAMM) from the github repository GitHub repository. It should be cloned into the "model" folder in the "<>.


Model checkpoints:
You can point the add-on to your trained model checkpoint folder using the file search option at the top of the "Face Model" pane.
The checkpoint directroy is expected to contain the following assets:
- checkpoint.pth
- config.yaml
- displacement_stats.pkl
- gaussian_id.pkl
- mean.obj
- mean_std_model.pkl

A sample config file has been provided in this repository.



Set-up
------
Once the plug-in has been installed and the model downloaded, you will need to installed the python libraries required to let it run. A script named ``setup.py'' has been provided for this purpose.

- Open setup.py in the Blender scripting pane
- Run the script and wait for all packages to install
- You may need to restart Blender after this step


Acknowledgements
----------------
If you found this work interesting and would like to use it in future projects, please cite:
(insert bibtex for the paper here)

