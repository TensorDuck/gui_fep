gui fep
=======

This is a package for analyzing a set of diffusion coordinates and selecting specific frames based upon their free energy in a visual manner
 
gui_fep.py
----------
 
Currently, all the useful stuff is contained here. Right now, this can do:
 
1. Load in a .ev (eigenvector) file from the LSDMap package
2. Select frames using the --interactive flag using a GUI
3. Save the frames to the file frames.ndx by pressing 'c'
4. Save the bounds for those frames to the file frames-info.txt
 
 
Basic Instructions for Quickstart
---------------------------------
 
Specify the file you want to load with the --dc_file 

The first two (non-trivial) DCs are plotted 

Use --interactive if you want to access the GUI for saving frames

Use the --append if you want to add the frames you find to the file. Otherwise, the files will be overwritten each time

--temp specifies the temperature. Useful for getting the absolute value of the bins in the plot. Please specify in Kelvin


Using the Interactive Feature
-----------------------------

When using the interactive feature, here are the commands you should know to use:
'x' symbolizes pressing the x button on your keyboard

'1' - Activites the box select. This lets you click and drag a box around the area you want to save frames to

'2' - Actives the bin select. This lest you click and drag a box around the area. Any bins touching or inside the box is saved.

'c' - Copies the frames to the file

'x' - Deactivites all the selection tools. Pressing three times to exit the interactive mode and close all the files when you are done.


Known Bugs and Future Fixes
---------------------------
1. Move classes into separate files for ease of use.
2. Currently, the built in GUI features from matplotlib can interfere with these added features. Plan to disable them in the future.
3. Implement Multi Select (select multiple bins in an irregular fashion) and circular select (select a circular or ellipsoidal area)
4. Option to select the group name of your choice when saving the frames
5. Handling multiple file types and formats
6. Handling weighted data


Please feel free to notify me of features you would find interseting or bugs you find annoying.
