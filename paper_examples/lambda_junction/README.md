To run the simulation:

- Run the file `main.py`.

To visualize the hydraulic heads:

- Go to the `out` folder. Here, you will find several `VTU` and `PVD` files 
  corresponding to the scheduled times. Open `ParaView` or similar, and import the 
  files to visualize them.

To visualize the water volume evolution:

- When you ran the simulation, a pickle file is generated inside the `out` folder. 
  To generate the plot for the water evolution inside the fracture, simply run the 
  `plot_water_volume.py` file.