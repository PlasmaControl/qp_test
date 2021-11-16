Minimal extraction + bug fixing of the box-constraint qp solver in the PCS from VEP, which relies on nstx_math. 

Before anything, be sure to 

`module load gcc7/default`

Compile with 

`gcc -D VEPN=2 -o main.exe main.c qp_solver.c nstx_math.c`

The defining of the macro VEPN=2 in commandline is because the code is organized to pass around multidimensional arrays, and everywhere in the code needs to know how big it is. Technically the nstx_math stuff is more general in the PCS but we only need square matrices. Perhaps this way of doing things is clunky, but from Stackoverflow/etc it seems this is just a problem with how C is made (and in the PCS there are a bunch of other macro definitions and background stuff I think). 

Run with 

`./main.exe`

For info see the [VEP spec](https://docs.google.com/document/d/1Qhbk7MiGolmaRhRa7-8sl-yOGcHsVb9RWQPyULVm31s/edit#) and in the PCS code vep_master.h and nstx_math.h
