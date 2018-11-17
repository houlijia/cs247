Raziel Haimi-Cohen

                   JPEG_Q_3017 - Quantization and Coding for Compressive
                   Sensing Measurements of Images.

1. Package Content

   The package consists of MATLAB and C/C++ Mex programs. It consists of
   source files (MATLAB files, C/C++ source files, Makefile-s, and other
   auxiliary files) and generated files (executables, objects, data files,
   etc.). All the generated files can be generated from the source files (with
   the exception of the MEX file 'myfwht'- see below).

   The source files are under version control. If you have access to our
   Subversion system you can get the source files by checking out the tree
   under compsens/releases/JPEG_Q_3017. It is a subst of a larger tree which
   is under compsens/trunk in version 3017. The package probably contains many more
   files than is actually needed because extracting just those files which are
   really needed would require too much work (changes in Makefiles, and other changes).

2. Building the generated files

   In order to build the generated files you have to run make in the ROOT
   directory (the directory in which this README file is in). Under Windows,
   run make from a Cygwin terminal window. There are three types of generated
   files:
 
   MEX files: MATLAB/MEX executables, objects, libraries and other auxiliary
     files.

   TST files: Normal C/C++ and Cuda executables, libraries and other auxiliary
     files.

   MLSEQ files: Those are data files (*.dat) which are stored in the MLSEQ_DIR
     directory (specified later). They are not necessary for this application but the makefile
     builds them anyway. 

   In addition the program needs the MEX file myfwht, which computes dyadic Walsh Hadamard
     Transform on arrays of type double. I do not have the source code for it. I have only the MEX
     executable for Windows, which is in ROOT/MPEG_quant.

   The MEX and TST programs can be built in different configurations. Each
   coniguration has a name which can be specified when running make, by
   specifying CFG=<configuration_name>, e.g.
                  make CFG=dbg
   will create the 'dbg' configuration. By default the 'std' configuration is
   built. The available configurations are
     
     std: Optimized, no assert checking.
     dbg: Not-optimized, contains debugging information and assert checking is
          turned on.
     timers: Like 'std', but enables various timers which print their output
       at the end of the execution.

   The make process checks if a GPU is available (more precisely, it checks
   the NVidia compiler, nvcc, is in the path), and if it is it will build
   programs which use the GPU. You can explicitly control it by appending '-g'
   or '-ng' to the configuration name to force it to use or not to use a
   GPU. Thus, running make with CFG=std-g will attempt to build the 'std'
   configuratioh using a GPU (and fail if a GPU is not available); running
   make with CFG=dbg-ng will build the 'dbg' version without using a GPU, even
   if a GPU is available.

   The make program puts the generated files for a specific configuration in
   directories which are specific to the configuration and the architecture

   MEX files are put in the directory ROOT/mex/CFG/MEX_ARCH, where CFG is the
     name of the configuration which was used and MEX_ARCH is a string
     describing the hardware architecture, the operating system and the Matlab
     version. Under Windows it also has the suffix '-vc' or '-gw', indicating
     that the compilation was done by the Visual C++ or by the MinGW
     compilers, respectively. e.g. 'x86_64-Linux-R2016a' or
     'x86_64-CYGWIN_NT-6.1-R2016a-vc'. The executable are in
     ROOT/mex/CFG/MEX_ARCH/exe, the libraries are under
     ROOT/mex/CFG/MEX_ARCH/lib and all other files are under
     ROOT/mex/CFG/MEX_ARCH/obj.

   TST files are put in the directory ROOT/tst/CFG/TST_ARCH, where CFG is the
     name of the configuration which was used and TST_ARCH is a string
     describing the hardware architecture and the operating
     system. e.g. 'x86_64-Linux' or 'x86_64-CYGWIN_NT-6.1'. The executable are
     in ROOT/tst/CFG/TST_ARCH/exe, the libraries are under
     ROOT/tst/CFG/TST_ARCH/lib and all other files are under
     ROOT/tst/CFG/TST_ARCH/obj.

   MLSEQ files are placed in the MLSEQ_DIR directory. By default, the MLSEQ_DIR
     is ROOT/../../../mlseq. If this place is not convenient override it by
     specifying the desired place, e.g.  make MLSEQ_DIR=my_mlseq

   The build process can be quite long (10-20 minutes if GPU is involved). You
   can speed it up considerably by using the -j option which parallelize the
   make processing.

3. Setting Matlab path

   Before starting the program we have to set the MATLAB path, making sure
   that it includes MEX files for the appropriate configuration and
   architecture. For that end do the following steps in MATLAB Command Window:
     1. Chage dirctory to ROOT/scripts
     2. Run 
            set_mex_path CFG [MEX_ARCH]
        where CFG is the configuration name and MEX_ARCH is the MEX_ARCH
        string. MEX_ARCH may be omitted if there is exactly one possible
        MEX_ARCH subdirectory. If you omitted MEX_ARCH and there is more than
        one possible values the command will produce an error and list the
        possible architectures.
     3. Check that the architecture is correct by running 
          get_mex_path

     Example:

              >> cd JPEG_Q_3017/scripts
              >> set_mex_path dbg-ng
              possible architectures:
              x86_64-CYGWIN_NT-6.1-R2016a-vc
              x86_64-Linux-R2016a
              Error using set_mex_path (line 89)
              Too many possible architectures
               
              >> set_mex_path dbg-ng x86_64-Linux-R2016a
              >> get_mex_path
              
              ans = 
              
                  'mex/dbg-ng/x86_64-Linux-R2016a/exe:'
              
              >> 

4. Running The Programs

   The main programs are in the subdirectory ROOT/JPEG_quant. All of them are
   MATLAB functions, i.e. they expect input arguments and return output
   arguments. You can get explanations about each of them by running 
      help <prog name>
   from the MATLAB Command Window, which prints the comment in the beginning
   of the function. Those functions are:

     getImageCodingInfo: Returns a struct which contains various parameters to
       be used by the encoder and the decoder.

     CSQuantize: Quantize a measurements vector and return an object
       containing the quantized measurements. Optionally write the object into
       a file in a compressed form.

     CSUnquantize: Receive a file containing compressed quantized measurements
       or an object containing quantized measurement and generate the vector
       of unquantized measurements.

     test_quant: Runs an end-to-end test. Gets and image file, generates
       measurements for it, calls getImageCodingInfo to get coding parameters,
       calls CSQuantize() and CSUnquantize to quantize and unquantize the
       measurements and print statistics on the number of bits used and the
       quantization error.

