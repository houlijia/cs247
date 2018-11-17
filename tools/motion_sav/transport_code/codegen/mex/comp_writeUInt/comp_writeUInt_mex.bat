@echo off
set MATLAB=C:\PROGRA~1\MATLAB\R2013b
set MATLAB_ARCH=win64
set MATLAB_BIN="C:\Program Files\MATLAB\R2013b\bin"
set ENTRYPOINT=mexFunction
set OUTDIR=.\
set LIB_NAME=comp_writeUInt_mex
set MEX_NAME=comp_writeUInt_mex
set MEX_EXT=.mexw64
call mexopts.bat
echo # Make settings for comp_writeUInt > comp_writeUInt_mex.mki
echo COMPILER=%COMPILER%>> comp_writeUInt_mex.mki
echo COMPFLAGS=%COMPFLAGS%>> comp_writeUInt_mex.mki
echo OPTIMFLAGS=%OPTIMFLAGS%>> comp_writeUInt_mex.mki
echo DEBUGFLAGS=%DEBUGFLAGS%>> comp_writeUInt_mex.mki
echo LINKER=%LINKER%>> comp_writeUInt_mex.mki
echo LINKFLAGS=%LINKFLAGS%>> comp_writeUInt_mex.mki
echo LINKOPTIMFLAGS=%LINKOPTIMFLAGS%>> comp_writeUInt_mex.mki
echo LINKDEBUGFLAGS=%LINKDEBUGFLAGS%>> comp_writeUInt_mex.mki
echo MATLAB_ARCH=%MATLAB_ARCH%>> comp_writeUInt_mex.mki
echo BORLAND=%BORLAND%>> comp_writeUInt_mex.mki
echo OMPFLAGS= >> comp_writeUInt_mex.mki
echo OMPLINKFLAGS= >> comp_writeUInt_mex.mki
echo EMC_COMPILER=msvc100>> comp_writeUInt_mex.mki
echo EMC_CONFIG=optim>> comp_writeUInt_mex.mki
"C:\Program Files\MATLAB\R2013b\bin\win64\gmake" -B -f comp_writeUInt_mex.mk
