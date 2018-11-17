# Get default hardware from 'uname -m'
ifeq ($(TARGET_HW),)
TARGET_HW:=$(shell uname -m )
endif

# Get default OS from 'uname -s'
ifeq ($(TARGET_OS),) 
TARGET_OS:=$(shell uname -s )
endif

ifeq ($(origin MEXCC),undefined)
ifeq ($(TARGET_OS),Linux)

MEXCC:=mex
MEX_COMPILER_EXT:=
MEX_COMPILER:=gcc

else

ifeq ($(COMSPEC),)
$(error Unknown OS) # Neither Linux nor Windows
endif

# determine MEX compiler

MEXCC:=mex.bat

ifneq ($(findstring Visual,$(shell $(MEXCC) -setup C | sed '1q')),)  # Visual C++ case

ifeq ($(findstring Visual,$(shell $(MEXCC) -setup C++ | sed '1q')),)
$(error different C and C++ Mex compilers)
endif

MEX_COMPILER:=MSVC
MEX_COMPILER_EXT:=-vc

else    # ifneq ($(findstring Visual,$(shell $(MEXCC) -setup C | sed '1q')),)

ifneq ($(findstring MinGW,$(shell $(MEXCC) -setup C | sed '1q')),) # MinGW case

ifeq ($(findstring MinGW,$(shell $(MEXCC) -setup C++ | sed '1q')),)
$(error different C and C++ Mex compilers)
endif

MEX_COMPILER:=MinGW
MEX_COMPILER_EXT:=-gw

else   # ifneq ($(findstring MinGW,$(shell $(MEXCC) -setup C | sed '1q')),)

$(error Unrecognized MEX compiler: $(shell $(MEXCC) -setup C | sed '1q'))

endif    # ifneq ($(findstring MinGW,$(shell $(MEXCC) -setup C | sed '1q')),)

endif	# ifneq ($(findstring Visual,$(shell $(MEXCC) -setup C | sed '1q')),)

endif   # ifeq ($(TARGET_OS),Linux)

# Compute matlab root
ifeq ($(origin MATLAB_PATH),undefined)

ifeq ($(strip $(shell which $(MEXCC) 2>/dev/null)),)

MEXCC:=
MATLAB_PATH:=
MEX_COMPILER:=
MEX_COMPILER_EXT:=

else

MATLAB_PATH:=$(shell which $(MEXCC))
MATLAB_PATH:=$(realpath $(MATLAB_PATH))

ifneq ($(findstring CYGWIN,$(TARGET_OS)),)
MATLAB_PATH:= $(shell cygpath -m "$(MATLAB_PATH)")
endif 

MATLAB_PATH:=$(shell echo "$(MATLAB_PATH)"| sed 's|/bin/[^/]*$$||')

endif  # ifeq ($(strip $(shell which $(MEXCC) 2>/dev/null)),)

export MATLAB_PATH
endif  # ($(origin MATLAB_PATH),undefined)

ifeq ($(TARGET_OS),Linux)
ifneq ($MEX_GCC_PATH),)
MEXCC:=export PATH="$$MEX_GCC_PATH:$$PATH"; $(MEXCC) 
endif
endif

endif 				# ifeq ($(origin MEXCC),undefined)

ifeq ($(ROOT_DIR),)
export ROOT_DIR:=$(realpath $(patsubst %/,%,$(shell cd ..; pwd)))
endif

ifeq ($(CFG),)
CFG:=std
endif

# Setting directories
# - special subdirectories
CUR_SUBDIR:=$(patsubst $(ROOT_DIR)/%,%,$(realpath $(CURDIR)))

ifeq ($(origin ARCH_SUBDIR),undefined)

export ARCH_SUBDIR:=$(CFG)/$(TARGET_HW)-$(TARGET_OS)

export TST_DIR:=$(ROOT_DIR)/tst/$(ARCH_SUBDIR)
$(info TST_DIR=$(TST_DIR))

ifneq ($(MATLAB_PATH),)

ifeq ($(TARGET_OS),Linux)
export MATLAB_VER=$(shell matlab -nodisplay -nojvm -r 'ver;exit' | sed -n \
  '/^[\t ]*MATLAB[\t ]*[Vv]ersion[\t ]*:/s/.*[:][^(]*[(]\([^)]*\)[)].*$$/\1/p;d')
else

export MATLAB_VER:=$(lastword $(subst /, ,$(MATLAB_PATH)))

endif

export MEX_DIR:=$(ROOT_DIR)/mex/$(ARCH_SUBDIR)-$(MATLAB_VER)$(MEX_COMPILER_EXT)

$(info MEX_DIR=$(MEX_DIR))

endif				# ifneq ($(MATLAB_PATH),)

endif 				# ifeq ($(origin ARCH_SUBDIR),undefined)

