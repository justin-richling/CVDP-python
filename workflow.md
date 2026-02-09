##

### NCL CVDP Output for Comparisons

/glade/work/richling/CVDP-LE/dev/cvdp

### Sample CVDP Created Data

/glade/derecho/scratch/richling/cvdp-output/netcdf_ensemble_new/


### Justin's Current Workflow

This workflow is subject to change, but this is the path necessary to match the graphics creation.

NOTES:
* It does currently save files and use those
* This is running via command line. This is introducing some env problems, like it still looks at `io.py` and does not see the CVDP module, but Python's. This will need to be addressed!
* There is a certain amount of redundancy, especially in the calculations/file saving/attributes. This is a great area to start looking at streamlining.
* This workflow now creates global plots for ensembles and ensemble averages (only global, no other plot types yet)
* The file saving is not ideal; the top level files are being written, but issues reading them before calcs, so these are being made everytime
* The climatologies are being checked, and are read in if exists, BUT the individual calculations are still being done everytime in `AtmOcnGR.py`:
    - NPI
    - EOF's (NAM, SAM, etc.)
    - "trends"
    - ensemble means
    - etc.

How to run:

1) activate CVDP development conda envrionment: `conda activate cvdp-dev` (must build first if haven't yet, see README.md)

2) Go to directory `cvdp`, and run: `python cli.py <directory for saved images>`, that's it.



This has a rigid structure, rife with loops. Obviously this is sped-built code that could benefit from refactoring and could better thought out. 