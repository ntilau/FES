# Finite Element Software a.k.a. FES
Code developed during my Ph.D., based on Armadillo NICTA, GetFEM Gmm++, METIS, MUMPS, TETGEN, ARPACK, OpenBlas, LAPACK and VTK postprocessing done with ParaView
Validations of the implementation was done through HFSS v10 mesh extraction and results comparisons, opening the door to the investigation of nonlinear and domain decomposition formulations

# How-to
To get started, clone the repository. Then run Maker.bat in windows terminal and type:
- `mingw32-make` to start compilation; 
- `mingw32-make test` for testing 

Alternatively, you can use WSL (Ubuntu) to cross-compile the windows binary; see the 'configure' file.

# Last Release
- Download [x86_64-w64-mingw32](https://github.com/ntilau/uni-phd-fes-cpp/raw/master/bin/FE.exe) build
- For testing, download [WR10](https://github.com/ntilau/uni-phd-fes-cpp/raw/master/bin/WR10_Prj.txt) project file and run `FE WR10 1e10`
