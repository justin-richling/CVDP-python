##

### NCL CVDP Output for Comparisons

/glade/work/richling/CVDP-LE/dev/cvdp

### Sample CVDP Created Data

/glade/derecho/scratch/richling/cvdp-output/netcdf_ensemble/


### Justin's Current Workflow

This workflow is subject to change, but this is the path necessary to match the graphics creation.

NOTES:
* It does currently save files and use those
* This is running via command line. This is introducing some env problems, like it still looks at `io.py` and does not see the CVDP module, but Python's. This will need to be addressed!
* There is a certain amount of redundancy, especially in the calculations/file saving. This is a great area to start looking at streamlining.
* This workflow now creates global plots for ensembles and ensemble averages (only global, no other plot types yet)
* 

How to run:

1) activate CVDP development conda envrionment: `conda activate cvdp-dev` (must build first if haven't yet, see README.md)

2) Go to directory `cvdp`, and run: `python cli.py <directory for saved images>`, that's it.