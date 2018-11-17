ifeq ($(ROOT_DIR),)
ROOT_DIR:=$(realpath $(patsubst %/,%,$(shell cd ..; pwd)))
endif
include $(ROOT_DIR)/util/basic_defs.mak

CC:=gcc
CPP:=g++

# Handle MEX definitions
ifneq ($(MATLAB_PATH),)

MEXCPP:=$(MEXCC)

ifeq ($(TARGET_OS),Linux)

MEX_CFLAGS=CFLAGS='$(filter-out -fPIC -Wall -Werror, $(CFLAGS)) -fPIC -Wall -Werror'

MEX_LIBS:=-lrt

MEX_EXT:=.mexa64
MEX_O_EXT:=.o
MEX_L_EXT:=.a

define MEX_LIB_CMD
@echo '==>' $(patsubst $(ROOT_DIR)/%,%,$@)
rm -f $@
ar -rcs $@ $(filter %$(MEX_O_EXT),$^)
endef

else    # ifeq ($(TARGET_OS),Linux)

ifeq ($(MEX_COMPILER),MSVC)

MEX_EXT:=.mexw64
MEX_O_EXT:=.obj
MEX_L_EXT:=.lib

MEX_CFLAGS=CFLAGS="$(CFLAGS) $(filter-out $(patsubst -w%,-w,$(filter -w%,$(CFLAGS))), -w4 -we4)"

MEX_LIBS:=

define MEX_LIB_CMD
@echo '==>' $(patsubst $(ROOT_DIR)/%,%,$@)
rm -f $@
lib /NOLOGO /WX /OUT:$(shell cygpath -m $@) $(shell cygpath -m $(filter %$(MEX_O_EXT),$^))
endef

else

ifeq ($(MEX_COMPILER),MinGW)

MEX_EXT:=.mexw64
MEX_O_EXT:=.obj
MEX_L_EXT:=.lib

MEXCC:=$(MEXCC)   -D_WIN32_WINNT=_WIN32_WINNT_WIN7
MEXCPP:=$(MEXCPP) -D_WIN32_WINNT=_WIN32_WINNT_WIN7

MEX_CFLAGS=CFLAGS="$$CFLAGS -Wall -Werror"

MEX_LIBS:=

define MEX_LIB_CMD
@echo '==>' $(patsubst $(ROOT_DIR)/%,%,$@)
rm -f $@
ar -rcs $@ $(filter %$(MEX_O_EXT),$^)
endef

else   # ifeq ($(MEX_COMPILER),MinGW)

$(error Unrecognized MEX compiler: $(shell $(MEXCC) -setup C | sed '1q'))

endif    # ifeq ($(MEX_COMPILER),MinGW)

endif    # ifneq ($(findstring Visual,$(shell $(MEXCC) -setup C | sed '1q')),)

endif   # ifeq ($(TARGET_OS),Linux)

ifeq ($(MEX_INCLUDE),)
# MEX_INCLUDE should be the include command to the path of the appropriate mex.h header file.
#  If it was not defind, it is determined by the following little shell script which
# takes the path to the current MEX compiler, and uses a SED script to
# determine include file. The sed script extracts the actual file from the
# symbolic link (if any) and folows Matlab's standard directory tree structure.

MEX_INCLUDE:=-I"$(MATLAB_PATH)"/extern/include
endif  # ifeq ($(MEX_INCLUDE),)

# Note that MEX automatically has -DNDEBUG
MEX_DEFINES:= -DMATLAB_MEX_FILE=1
MEX_OPTS=$(INCLUDE_OPT) $(MEX_DEFINES) -largeArrayDims -silent $(MEX_CFLAGS)

# MEX_OUTPUT:=-output $$@ -outdir $$(@D)

endif				# ifneq ($(MATLAB_PATH),)

ifeq ($(TARGET_OS),Linux)

OBJ_EXT:=.o
LIB_EXT:=.a
EXE_EXT:=

define TST_LIB_CMD
@echo '==>' $(patsubst $(ROOT_DIR)/%,%,$@)
rm -f $@
ar -rcs $@  $(filter %$(OBJ_EXT),$^)
endef

# This commented out code computes the GCC major and minor versions.
# we may need it someday.
#
# GCC_VER:=$(shell gcc --version | sed -n 's/^[^0-9]*\([0-9]*\)[.]\([0-9]*\)[.]..*/\1 \2/p;1q')
#
# GCC_VER_MAJOR:=$(word 1,$(GCC_VER))
# GCC_VER_MINOR:=$(word 2,$(GCC_VER))
# 
# ifneq ($(find $(GCC_VER_MAJOR),1 2 3),)
# $(warning GCC_MAJOR < 4 [$(find $(GCC_VER_MAJOR),1 2 3)])
# GCC_OLD:=YES
# else
# $(warning GCC_MAJOR >= 4 [$(find $(GCC_VER_MAJOR),1 2 3)])
# ifneq ($(find $(GCC_VER_MAJOR), 4),)
# $(warning GCC_MAJOR = 4 [$(find $(GCC_VER_MAJOR), 4)])
# ifneq ($(find $(GCC_VER_MINOR), 1 2 3 4),)
# GCC_OLD:=YES
# endif
# endif
# endif
# 
# $(warning GCC_VER = "$(GCC_VER)" $(GCC_VER_MAJOR) $(GCC_VER_MINOR) $(GCC_OLD))

else				# Assume it is Windows

ifeq ($(COMSPEC),)
$(error Unknown OS) # Neither Linux nor Windows
endif

define TST_LIB_CMD
@echo '==>' $(patsubst $(ROOT_DIR)/%,%,$@)
@rm -f $@
ar -rc -s $@ $(filter %$(OBJ_EXT),$^)
endef

OBJ_EXT:=.o
LIB_EXT:=.a
EXE_EXT:=.exe

endif				# End OF Windows

ifneq ($(filter kanas%,$(HOSTNAME)),)
OPENCV_LIBS:=-L/usr/local/lib -L/usr/lib \
-lopencv_calib3d -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui \
-lopencv_imgproc -lopencv_videoio -lopencv_ml -lopencv_objdetect -lopencv_photo \
-lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab
else
OPENCV_LIBS:=-L/usr/local/lib \
-lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann \
-lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree \
-lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts \
-lopencv_video -lopencv_videostab
endif

# For debugging purposes of the makefile. Running "make DUMP=" inserts at the
# beginning of each terminal output a prefix containing with place in the
# makefile in from which printing was done, and if this is a target building,
# the target and the dependencies that caused the building
ifneq ($(origin DUMP),undefined)
OLD_SHELL:=$(SHELL)
SHELL=$(warning [$@: $?])$(OLD_SHELL) -x
endif

CFG_DIR:=$(ROOT_DIR)/cfg
CFG_FILE:=$(CFG_DIR)/$(CFG).mak
include $(CFG_FILE)

# - Tst directories
TST_EXE_ROOT_DIR:=$(TST_DIR)/exe
TST_EXE_DIR:=$(TST_EXE_ROOT_DIR)
TST_LIB_DIR:=$(TST_DIR)/lib
TST_OBJ_DIR:=$(TST_DIR)/obj/$(CUR_SUBDIR)

TST_DIRS_USED=$(TST_EXE_DIR) $(TST_LIB_DIR) $(TST_OBJ_DIR)
DIRS_USED=$(MEX_DIRS_USED) $(TST_DIRS_USED)

BLD_TRGTS:= all exe preproc obj lib dpnd dirs
TST_BLD_TRGTS:=$(patsubst %,%-tst,$(BLD_TRGTS))

# Note that the clobber target here is not invoked by running 
# make clobber for $(ROOT_DIR).
DEL_TRGTS:=$(patsubst %,del-%,$(BLD_TRGTS)) clobber
TST_DEL_TRGTS:=$(patsubst %,%-tst,$(DEL_TRGTS))

ifneq ($(MATLAB_PATH),)
MEX_BLD_TRGTS:=$(patsubst %,%-mex,$(BLD_TRGTS)) preproc-opts-mex
MEX_DEL_TRGTS:=$(patsubst %,%-mex,$(DEL_TRGTS)) del-preproc-opts-mex
endif

TRGTS:=$(BLD_TRGTS) $(MEX_BLD_TRGTS) $(TST_BLD_TRGTS) \
    $(DEL_TRGTS) $(MEX_DEL_TRGTS) $(TST_DEL_TRGTS) \
    data del-data mex del-mex tst del-tst clean
.PHONY: $(TRGTS) do_data
all:

ifneq ($(MATLAB_PATH),)

# - Mex directories
MEX_EXE_ROOT_DIR:=$(MEX_DIR)/exe
MEX_EXE_DIR:=$(MEX_EXE_ROOT_DIR)
MEX_LIB_DIR:=$(MEX_DIR)/lib
MEX_OBJ_DIR:=$(MEX_DIR)/obj/$(CUR_SUBDIR)
MEX_DIRS_USED=$(MEX_EXE_DIR) $(MEX_LIB_DIR) $(MEX_OBJ_DIR)

PREPROC_OPTS_MEX_FILES:=$(patsubst %,$(MEX_OBJ_DIR)/%, mex_c_opts.out mex_cpp_opts.out mex_preproc.mak)

# Note that the processing includes
# converting '#' into '@@@' and back. This is because the path may containg
# F#, and make interpretes '#' as the start of a comment...
$(MEX_OBJ_DIR)/mex_preproc.mak: $(MEX_OBJ_DIR)/mex_c_opts.out $(MEX_OBJ_DIR)/mex_cpp_opts.out $(MAKEFILE)
	@echo '==>' $(patsubst $(ROOT_DIR)/%,%,$@)
	@rm -f $@
ifeq ($(TARGET_OS),Linux)
	@echo 'MX_C_OPTS_FILE:=$(shell sed -n \
		'/^[\t ]*Options[\t ]*file[\t ]*:/{s/^[\t ]*Options[\t ]*file[\t ]*:[\t ]*//;p;q}' \
                $(MEX_OBJ_DIR)/mex_c_opts.out)' >> $@
	@echo 'MX_CPP_OPTS_FILE:=$(shell sed -n \
		'/^[\t ]*Options[\t ]*file[\t ]*:/{s/^[\t ]*Options[\t ]*file[\t ]*:[\t ]*//;s/[\t ]*$$//;p;q}' $(MEX_OBJ_DIR)/mex_cpp_opts.out)' >> $@
	@echo 'MX_CC_PREPROC:= $(shell sed -n \
		'/^[\t ]*CMDLINE1[\t ]*:/{s/^[\t ]*CMDLINE1[\t ]*:[\t ]*//;s/[^\t ]*_dummy.c//;s/-o[\t ]*[^\t ]*_dummy$(MEX_O_EXT)//;s|[\t ][-/]c[\ ]| |;s|[\t ]-DNDEBUG[^\t ]*||g;s|$$| -E $$(MEX_PRPRC_OPTS) |p;q}' \
		$(MEX_OBJ_DIR)/mex_c_opts.out)' >> $@
	@echo 'MX_CPP_PREPROC:= $(shell sed -n \
		'/^[\t ]*CMDLINE1[\t ]*:/{s/^[\t ]*CMDLINE1[\t ]*:[\t ]*//;s/[^\t ]*_dummy.cc//;s/-o[\t ]*[^\t ]*_dummy$(MEX_O_EXT)//;s|[\t ][-/]c[\ ]| |;s|[\t ]-DNDEBUG[^\t ]*||g;s|$$| -E $$(MEX_PRPRC_OPTS) |p;q}' \
		$(MEX_OBJ_DIR)/mex_cpp_opts.out)' >> $@
else
	@echo 'MX_C_OPTS_FILE:=$(shell sed -n \
		'/^[\t ]*Options[\t ]*file[\t ]*:/{s/^[\t ]*Options[\t ]*file[\t ]*:[\t ]*//;s/[\t ]*$$//;s/[\t ]*$$//;s|\\|/|g;p;q}'\
                $(MEX_OBJ_DIR)/mex_c_opts.out)' >> $@
	@echo 'MX_CPP_OPTS_FILE:=$(shell sed -n \
		'/^[\t ]*Options[\t ]*file[\t ]*:/{s/^[\t ]*Options[\t ]*file[\t ]*:[\t ]*//;s/[\t ]*$$//;s|\\|/|g;p;q}' $(MEX_OBJ_DIR)/mex_cpp_opts.out)' >> $@
	@echo 'MX_CC_INC:=$(shell sed -n '/^ *Set INCLUDE *=/{s/^ *Set INCLUDE *=//;s/^ */-I "/;s/;;*[ ]*$$/"/;s/;;*/" -I "/g;s|\\|/|gp}' \
		$(MEX_OBJ_DIR)/mex_c_opts.out)' >> $@
	@echo 'MX_CPP_INC:=$(shell sed -n '/^ *Set INCLUDE *=/{s/^ *Set INCLUDE *=//;s/^ */-I "/;s/;*[\t ]*$$/"/;s/;;*/" -I "/g;s|\\|/|gp}' \
		 $(MEX_OBJ_DIR)/mex_cpp_opts.out)' >> $@
	@echo 'MX_CC_PATH_CMD:=$(shell sed  -n \
		'/^ *Set PATH *= */{s/^ *Set PATH *= *//;s|^\([a-zA-Z]\):|/cygdrive/\1|;s|;\([a-zA-Z]\):|;/cygdrive/\1|g;s|\\|/|g;s/;/:/g;s/\#/@@@/;s/^/export PATH="/;s/:*"*$$/"/p}' \
		$(MEX_OBJ_DIR)/mex_c_opts.out)' >> $@
	@echo 'MX_CPP_PATH_CMD:=$(shell sed -n \
		'/^ *Set PATH *= */{s/^ *Set PATH *= *//;s|^\([a-zA-Z]\):|/cygdrive/\1|;s|;\([a-zA-Z]\):|;/cygdrive/\1|g;s|\\|/|g;s/;/:/g;s/\#/@@@/;s/^/export PATH="/;s/:*"*$$/"/p}' \
		 $(MEX_OBJ_DIR)/mex_cpp_opts.out)' >> $@
	@echo 'MX_CC_PREPROC:= $$(MX_CC_PATH_CMD); $(shell sed -n \
		'1,/^Building/d;s/[^\t ]*_dummy.c//;s/[-/][^\t ]*[\t ]*_dummy$(MEX_O_EXT)//;s|^\([a-zA-Z]\):|/cygdrive/\1|;s|\\|/|g;s|[\t ][-/]c[\ ]| |;s|[\t ]-DNDEBUG[^\t ]*||g;s|$$| -E $$(MX_CC_INC) $$(MEX_PRPRC_OPTS) |p;q' \
		$(MEX_OBJ_DIR)/mex_c_opts.out)' >> $@
	@echo 'MX_CPP_PREPROC:= $$(MX_CPP_PATH_CMD); $(shell sed -n \
		'1,/^Building/d;s/[^\t ]*_dummy.cc//;s/[-/][^\t ]*[\t ]*_dummy$(MEX_O_EXT)//;s|^\([a-zA-Z]\):|/cygdrive/\1|;s|\\|/|g;s|[\t ][-/]c[\ ]| |;s|[\t ]-DNDEBUG[^\t ]*||g;s|$$| -E $$(MX_CPP_INC) $$(MEX_PRPRC_OPTS) |p;q' \
		 $(MEX_OBJ_DIR)/mex_cpp_opts.out)' >> $@
endif

$(MEX_OBJ_DIR)/mex_c_opts.out: 
	@echo '==>' $(patsubst $(ROOT_DIR)/%,%,$@)
	@cd $(@D); echo "" > _dummy.c;  $(MEXCC)  -c -v -n _dummy.c  > $(@F); rm -f _dummy.c

$(MEX_OBJ_DIR)/mex_cpp_opts.out:
	@echo '==>' $(patsubst $(ROOT_DIR)/%,%,$@)
	@cd $(@D); echo "" > _dummy.cc; $(MEXCPP) -c -v -n _dummy.cc > $(@F); rm -f _dummy.cc

$(patsubst %,$(MEX_OBJ_DIR)/%,mex_c_opts.out mex_cpp_opts.out): $(filter-out $(wildcard $(MEX_OBJ_DIR)),$(MEX_OBJ_DIR))

ifeq ($(DO_M_DPND),YES)

MEX_PRPRC_OPTS:=$(INCLUDE_OPT) $(MEX_DEFINES)
include $(MEX_OBJ_DIR)/mex_preproc.mak

MEX_CC_PREPROC:=$(subst @@@,\#,$(MX_CC_PREPROC))
MEX_CPP_PREPROC:=$(subst @@@,\#,$(MX_CPP_PREPROC))
MEXCC+= -f $(MX_C_OPTS_FILE)
MEXCPP+= -f $(MX_CPP_OPTS_FILE)

#  PPATH="$$PATH"; $(MEX_CC_PATH_CMD) ; \
#  PPATH="$$PATH"; $(MEX_CPP_PATH); \
#  $(MEX_OPTS) $($*_OPTS) $(MEX_DEFINES) $(MEX_INCLUDE)
#  $(MEX_OPTS) $($*_OPTS) $(MEX_DEFINES) $(MEX_INCLUDE)
#

endif   			# ifeq ($(DO_M_DPND),YES)

endif				# ifneq ($(MATLAB_PATH),)


all:				# Make this the default target

# Commands for computing dependencies. The commands are generated by call(),
# with the following argumentes:
#   $(1) The processing command, including specific options, e.g. "$(CC)"
#        $(CC_OPTS).
#   $(2) Source file filter, e.g. "%.c %.cu"
#   $(3) dependency file extension, e.g. ".dpn".
#   $(4) target file substitutions for %$(3), (may be more than one), e.g. 
#        "%.o %.i" or "%$(EXE_EXT)."
#  \note When invoking the call there should be no space betwee $(3) and the
#        preceding comma, as in the examples below.

# These variables contain parts of a SED script to extract dependencies from a preprocessed file
# - /^#.*"[^<].*"/!d;/^#pragma/d; select only the line containing file
#     names. These are lines which begin with '#' and contain a file
#     name in quotes Ignore filenames
#     beginning with '<' and lines beginning with #pragma.
# - s|^.*"\(.*\)".*$$|\1|;  extract the file name
DPND_SED_1:=/^\#.*"[^<].*"/!d;/^\#pragma/d;s|^.*"\(.*\)".*$$|\1|

# - s|^.*"\(.*\)".*$$|\1|;  extract the file name
# - s|\\\\|/|g;s|\\|/|g; replace backslash file separator by slashes
#     (sometimes the backslash separators appear as a pair x\\y).
# - s|^\([A-Za-z]\):/|/cygdrive/\1/|; Replace Windows drive ( C:/ ) by
#     Cygwin drive (/cygdrive/C/).
DPND_SED_W:=;s|\\\\|/|g;s|\\|/|g;s|^\([A-Za-z]\)[:]|/cygdrive/\1|

# - \|/[\t ]*$$|d; Ignore file names ending with / (directory name)
# - \|^$(ROOT_DIR)|I{p;d};\|^$(realpath $(ROOT_DIR))|I{p;d};/^[^/]/p Print only relative path files or
#     files in the $(ROOT_DIR) tree. Note that the match here uses 'I' to be case insensitive
DPND_SED_2:=;\|/[\t ]*$$|d;s/$$/ \\/;\|^$(ROOT_DIR)|I{p;d};\|^$(realpath $(ROOT_DIR))|I{p;d};/^[^/]/p

ifneq ($(findstring CYGWIN,$(TARGET_OS)),) # Cygwin case. Handle Windows style file names

# The sed script had the following functions:
# - /^#.*"[^<].*"/!d;/^#pragma/d; select only the line containing file
#     names. These are lines which begin with '#' and contain a file
#     name in quotes Ignore filenames
#     beginning with '<' and lines beginning with #pragma.
# - s|^.*"\(.*\)".*$$|\1|;  extract the file name
# - s|\\\\|/|g;s|\\|/|g; replace backslash file separator by slashes
#     (sometimes the backslash separators appear as a pair x\\y).
# - s|^\([A-Za-z]\):/|/cygdrive/\1/|; Replace Windows drive ( C:/ ) by
#     Cygwin drive (/cygdrive/C/).
# - \|/[\t ]*$$|d; Ignore file names ending with / (directory name)
# - \|^$(ROOT_DIR)|I{p;d};\|^$(realpath $(ROOT_DIR))|I{p;d};/^[^/]/p Print only relative path files or
#     files in the $(ROOT_DIR) tree. Note that the match here uses 'I' to be case insensitive
define DPND_OBJ_TMPLT
@echo "$(<F) --> $(patsubst $(ROOT_DIR)/%,%,$@)"
rm -f $@ $@.err;
echo '$@ $(foreach e, $(2), $(patsubst %$(1),$(e),$@)) :' | sed 's|$$| \\|' > $@- ;
sed -n '$(DPND_SED_1)$(DPND_SED_W)$(DPND_SED_2)' $< | sort -u  >> $@- ; \
  if [ ! -s $@.err ]; then mv $@- $@;  rm -f $@.err; \
  else echo "error in building $@"; cat $@.err; exit 1 ; fi;
endef

else  # ifneq ($(findstring CYGWIN,$(TARGET_OS)),)

define DPND_OBJ_TMPLT
@echo "$(<F) --> $(patsubst $(ROOT_DIR)/%,%,$@)"
rm -f $@ $@.err;
echo '$@ $(foreach e, $(2), $(patsubst %$(1),$(e),$@)) :' | sed 's|/*[\t ]*$$| \\|' > $@- ;
sed -n '$(DPND_SED_1)$(DPND_SED_2)' $< 2>$@.err | sort -u  >> $@- ; \
  if [ ! -s $@.err ]; then mv $@- $@;  rm -f $@.err; \
  else echo "error in building $@"; cat $@.err; exit 1 ; fi;
endef

endif # ifneq ($(findstring CYGWIN,$(TARGET_OS)),)

ifneq ($(MATLAB_PATH),)

define MEX_CC_OBJ_DPND_CMD
$(call DPND_OBJ_TMPLT,.odpn,%.i %$(MEX_O_EXT))
endef

define MEX_CPP_OBJ_DPND_CMD
$(call DPND_OBJ_TMPLT,.odpn,  %.i %$(MEX_O_EXT))
endef

endif      # ifneq ($(MATLAB_PATH),)

define CC_OBJ_DPND_CMD
$(call DPND_OBJ_TMPLT,.odpn, %.i %$(OBJ_EXT))
endef

define CPP_OBJ_DPND_CMD
$(call DPND_OBJ_TMPLT,.odpn, %.i %$(OBJ_EXT))
endef

define PREPROC_TMPLT
@echo "$(<F) --> $(patsubst $(ROOT_DIR)/%,%,$@)"
rm -f $@ $@.err;
$(1) -E $(filter $(2),$^) > $@- 2>$@.err; \
  if [ ! -s $@.err ]; then mv $@- $@;  rm -f $@.err; \
  else echo "error in building $@"; cat $@.err; exit 1 ; fi; 
endef

ifneq ($(MATLAB_PATH),)

ifneq ($(MEX_COMPILER),MSVC)

define MEX_CC_PREPROC_CMD
$(call PREPROC_TMPLT, $(MEX_CC_PREPROC),%.c)
endef

define MEX_CPP_PREPROC_CMD
$(call PREPROC_TMPLT, $(MEX_CPP_PREPROC),%.c %.cc)
endef

else				# ifneq ($(MEX_COMPILER),MSVC)

# Special handling of MSVC compiler. In preprocessing it writes the name of
# each processed file to stderr

define MEX_CC_PREPROC_CMD
@echo "$(<F) --> $(patsubst $(ROOT_DIR)/%,%,$@)"
rm -f $@ $@.err;
$(MEX_CC_PREPROC) $< > $@- 2>$@.err-; sed '1d' $@.err- > $@.err; rm $@.err-; \
  if [ ! -s $@.err ]; then mv $@- $@;  rm -f $@.err; \
  else echo "error in building $@"; cat $@.err; exit 1 ; fi; 
endef

define MEX_CPP_PREPROC_CMD
@echo "$(<F) --> $(patsubst $(ROOT_DIR)/%,%,$@)"
rm -f $@ $@.err;
$(MEX_CPP_PREPROC) $< > $@- 2>$@.err-; sed '1d' $@.err- > $@.err; rm $@.err-; \
  if [ ! -s $@.err ]; then mv $@- $@;  rm -f $@.err; \
  else echo "error in building $@"; cat $@.err; exit 1 ; fi; 
endef

endif				# ifneq ($(MEX_COMPILER),MSVC)

endif   # ifneq ($(MATLAB_PATH),)

define CC_PREPROC_CMD
$(call PREPROC_TMPLT, $(CC) -E $(CC_OPTS) $($*_OPTS),%.c)
endef

define CPP_PREPROC_CMD
$(call PREPROC_TMPLT, $(CPP) -E $(CPP_OPTS) $($*_OPTS),%.c %.cc)
endef

# Filter to select source files. Not 
ifeq ($(GPU),NO)
CU_EXT:=-!NO_GPU!  # A dummy value that should never appear
endif

# Fitering expression to select source files. Note that it is a a "recursively
# expandable" variable (set by '=' rather than':='), because if there is a GPU
# $(CU_EXT) is set later.
SRC_FLTR=%.c %.cc %$(CU_EXT)

ifneq ($(MATLAB_PATH),)

ifneq ($(findstring CYGWIN,$(TARGET_OS)),) # Cygwin case. mex output options does not work well

define MEX_CC_EXE_CMD
@echo '==>' $(patsubst $(ROOT_DIR)/%,%,$@)
$(MEXCC) $(MEX_OPTS) -output $(@F) \
  $(shell cygpath -m $(filter %$(SRC_FLTR) %$(MEX_O_EXT), $^ ) $(filter %$(MEX_L_EXT), $^)) $(MEX_LIBS) && \
  mv $(@F) $@ && rm -f $(@F).pdb
endef

define MEX_CPP_EXE_CMD
@echo '==>' $(patsubst $(ROOT_DIR)/%,%,$@)
$(MEXCPP) $(MEX_OPTS) -output $(@F) \
  $(shell cygpath -m $(filter %$(SRC_FLTR) %$(MEX_O_EXT), $^ ) $(filter %$(MEX_L_EXT), $^)) $(MEX_LIBS) && \
  mv $(@F) $@ && rm -f $(@F).pdb
endef

define MEX_CC_OBJ_CMD
@echo "$(<F) --> $(patsubst $(ROOT_DIR)/%,%,$@)"
$(MEXCC) -c $(MEX_OPTS) $(MEX_INCLUDE) $<  &&  mv $(@F) $@
endef

define MEX_CPP_OBJ_CMD
@echo "$(<F) --> $(patsubst $(ROOT_DIR)/%,%,$@)"
$(MEXCPP) -c $(MEX_OPTS) $(MEX_INCLUDE) $<  &&  mv $(@F) $@
endef

else   # Not Cygwin case

define MEX_CC_EXE_CMD
@echo '==>' $(patsubst $(ROOT_DIR)/%,%,$@)
$(MEXCC) $(MEX_OPTS) -outdir $(@D) -output $@ \
  $(filter $(SRC_FLTR) %$(MEX_O_EXT), $^ ) $(filter %$(MEX_L_EXT), $^) $(MEX_LIBS)
endef

define MEX_CPP_EXE_CMD
@echo '==>' $(patsubst $(ROOT_DIR)/%,%,$@)
$(MEXCPP) $(MEX_OPTS) -outdir $(@D) -output $@ \
  $(filter $(SRC_FLTR) %$(MEX_O_EXT), $^ ) $(filter %$(MEX_L_EXT), $^) $(MEX_LIBS)
endef

define MEX_CC_OBJ_CMD
@echo "$(<F) --> $(patsubst $(ROOT_DIR)/%,%,$@)"
$(MEXCC) -c $(MEX_OPTS) $(MEX_INCLUDE) -outdir $(@D) $<
endef

define MEX_CPP_OBJ_CMD
@echo "$(<F) --> $(patsubst $(ROOT_DIR)/%,%,$@)"
$(MEXCPP) -c $(MEX_OPTS) $(MEX_INCLUDE) -outdir $(@D) $<
endef

endif  # Cygwin or not

endif  # ifneq ($(MATLAB_PATH),)

define CC_EXE_CMD
@echo '==>' $(patsubst $(ROOT_DIR)/%,%,$@)
$(CC) $(CC_OPTS) $($*_OPTS) -o $@ $(filter $(SRC_FLTR) %$(OBJ_EXT), $^ ) $(filter %$(LIB_EXT), $^) \
  $(TST_CC_SYS_LIBS)
endef

define CPP_EXE_CMD
@echo '==>' $(patsubst $(ROOT_DIR)/%,%,$@)
$(CPP) $(CPP_OPTS) $($*_OPTS) -o $@ $(filter %$(OBJ_EXT) $(SRC_FLTR), $^ ) $(filter %$(LIB_EXT), $^) \
  $(TST_CPP_SYS_LIBS)
endef

define TST_CC_OBJ_CMD
@echo "$(<F) --> $(patsubst $(ROOT_DIR)/%,%,$@)"
$(CC) -c $(CC_OPTS) $($*_OPTS) $<  -o $@
endef

define TST_CPP_OBJ_CMD
@echo "$(<F) --> $(patsubst $(ROOT_DIR)/%,%,$@)"
$(CPP) -c $(CPP_OPTS) $($*_OPTS) $< -o $@
endef

# MEX_CXXFLAGS:= \
#  CXXFLAGS='-ansi -fexceptions -fPIC -fno-omit-frame-pointer -pthread -std=c++98'


ifneq ($(GPU),NO)

include $(ROOT_DIR)/util/cuda_defs.mak

endif

