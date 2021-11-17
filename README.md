Minimal extraction + bug fixing of the box-constraint qp solver in the PCS from VEP, which relies on nstx_math. 

Before anything, be sure to 

`module load gcc7/default`

Compile with 

`gcc -o main.exe main.c qp_solver.c nstx_math.c`

Run with 

`./main.exe`

For info see the [VEP spec](https://docs.google.com/document/d/1Qhbk7MiGolmaRhRa7-8sl-yOGcHsVb9RWQPyULVm31s/edit#) and in the PCS code vep_master.h and nstx_math.h

To instead build a shared library, use 
```
gcc -c -fPIC main.c
gcc -shared -fPIC -D VERBOSE_LEVEL=0 -o libqp_solver.so qp_solver.c nstx_math.c
```
where verbose level would otherwise default to 1, showing minimal output. 

You can use it with C via
```
gcc -o main.exe -L. main.o -lqp_solver
```

For python, after making the shared library, just call
```
python main.py
```
