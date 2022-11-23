Detailed installation guide: https://gmd.copernicus.org/articles/12/4955/2019/#section10

Prerequisites:
gfortran
jasper (http://www.ece.uvic.ca/~mdadams/jasper/)
eccodes (https://confluence.ecmwf.int/display/ECC/ecCodes+installation)
NetCDF-C (https://www.unidata.ucar.edu/downloads/netcdf/)
NetCDF-Fortran (https://www.unidata.ucar.edu/downloads/netcdf/)

Compiling:
Add the correct library paths to the makefile
At the bottom of the par_mod.f90 file, an array containing the names of the interpolated fields can be found:
	For the linear interplated windfields:
		filenames=(/ 'lin_wind_XXX.nc','                               ','                               ' /)
	Neural network interpolated windfields:
		filenames=(/ 'nn_wind_XXX.nc','                               ','                               ' /)
	Original windfields (not interpolated):
		filenames=(/ '                               ','                               ','                               ' /)
	XXX should be replaced by the timestamps of the filenames of the desired Season.
	Neural network interpolated velocity fields can be found in:
	 - Model 2: https://zenodo.org/record/7318809
	 - Model 4: https://zenodo.org/record/7277854
	Linear interpolated fields can be obtained by following the instructions in the README.md in https://zenodo.org/record/7065139
Run 'make' in the directory and it should compile an executable called 'FLEXPART'

Running FLEXPART:
	Modify the pathnames file:
		1st line: Add the correct directory path for the desired season (i.e. ./options_april for the April run)
		2nd line: Path to were the output data will be written to
		3rd line: Path to the ERA5 data
		4th line: Path to the AVAILABLE file (an example can be found in the source directory)
	Execute FLEXPART from the same directory as where the 'pathnames' file is.

	The particle data will appear in the output directory.

	
