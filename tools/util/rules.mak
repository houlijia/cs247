# remove duplicate items from source file lists
MEX_C_SRCS:=$(sort $(MEX_C_SRCS))
MEX_CPP_SRCS:=$(sort $(MEX_CPP_SRCS))
MEX_CUDA_SRCS:=$(sort $(MEX_CUDA_SRCS))
TST_C_SRCS:=$(sort $(TST_C_SRCS))
TST_CPP_SRCS:=$(sort $(TST_CPP_SRCS))
TST_CUDA_SRCS:=$(sort $(TST_CUDA_SRCS))

# executable files
TST_EXE_FILES:=$(TST_C_EXE_FILES) $(TST_CPP_EXE_FILES) $(TST_CUDA_EXE_FILES)
MEX_EXE_FILES:=$(MEX_C_EXE_FILES) $(MEX_CPP_EXE_FILES) $(MEX_CUDA_EXE_FILES)

# Object files
MEX_C_OBJ_FILES:=$(patsubst %.c,$(MEX_OBJ_DIR)/%$(MEX_O_EXT), $(MEX_C_SRCS))
MEX_CPP_OBJ_FILES:=$(patsubst %.cc,$(MEX_OBJ_DIR)/%$(MEX_O_EXT), $(MEX_CPP_SRCS))
MEX_CUDA_OBJ_FILES:=$(patsubst %.cu,$(MEX_OBJ_DIR)/%$(MEX_O_EXT), $(MEX_CUDA_SRCS))
TST_C_OBJ_FILES:=$(patsubst %.c,$(TST_OBJ_DIR)/%$(OBJ_EXT), $(TST_C_SRCS))
TST_CPP_OBJ_FILES:=$(patsubst %.cc,$(TST_OBJ_DIR)/%$(OBJ_EXT), $(TST_CPP_SRCS))
TST_CUDA_OBJ_FILES:=$(patsubst %.cu,$(TST_OBJ_DIR)/%$(OBJ_EXT), $(TST_CUDA_SRCS))
MEX_OBJ_FILES:=$(MEX_C_OBJ_FILES) $(MEX_CPP_OBJ_FILES) $(MEX_CUDA_OBJ_FILES)
TST_OBJ_FILES:=$(TST_C_OBJ_FILES) $(TST_CPP_OBJ_FILES) $(TST_CUDA_OBJ_FILES)

# Preproc files
MEX_C_PREPROC_FILES:=$(patsubst %.c,$(MEX_OBJ_DIR)/%.i, $(MEX_C_SRCS))
MEX_CPP_PREPROC_FILES:=$(patsubst %.cc,$(MEX_OBJ_DIR)/%.i, $(MEX_CPP_SRCS))
MEX_CUDA_PREPROC_FILES:=$(patsubst %.cu,$(MEX_OBJ_DIR)/%.i, $(MEX_CUDA_SRCS))
TST_C_PREPROC_FILES:=$(patsubst %.c,$(TST_OBJ_DIR)/%.i, $(TST_C_SRCS))
TST_CPP_PREPROC_FILES:=$(patsubst %.cc,$(TST_OBJ_DIR)/%.i, $(TST_CPP_SRCS))
TST_CUDA_PREPROC_FILES:=$(patsubst %.cu,$(TST_OBJ_DIR)/%.i, $(TST_CUDA_SRCS))

MEX_PREPROC_FILES:=$(MEX_C_PREPROC_FILES) $(MEX_CPP_PREPROC_FILES) $(MEX_CUDA_PREPROC_FILES)
TST_PREPROC_FILES:=$(TST_C_PREPROC_FILES) $(TST_CPP_PREPROC_FILES) $(TST_CUDA_PREPROC_FILES)

# Object dependency files
MEX_C_ODPN_FILES:=$(patsubst %.c,$(MEX_OBJ_DIR)/%.odpn, $(MEX_C_SRCS))
MEX_CPP_ODPN_FILES:=$(patsubst %.cc,$(MEX_OBJ_DIR)/%.odpn, $(MEX_CPP_SRCS))
MEX_CUDA_ODPN_FILES:=$(patsubst %.cu,$(MEX_OBJ_DIR)/%.odpn, $(MEX_CUDA_SRCS))
TST_C_ODPN_FILES:=$(patsubst %.c,$(TST_OBJ_DIR)/%.odpn, $(TST_C_SRCS))
TST_CPP_ODPN_FILES:=$(patsubst %.cc,$(TST_OBJ_DIR)/%.odpn, $(TST_CPP_SRCS))
TST_CUDA_ODPN_FILES:=$(patsubst %.cu,$(TST_OBJ_DIR)/%.odpn, $(TST_CUDA_SRCS))

MEX_ODPN_FILES:=$(MEX_C_ODPN_FILES) $(MEX_CPP_ODPN_FILES) $(MEX_CUDA_ODPN_FILES)
TST_ODPN_FILES:=$(TST_C_ODPN_FILES) $(TST_CPP_ODPN_FILES) $(TST_CUDA_ODPN_FILES)

# Rules for directory generation
DIRS_NEEDED:=$(filter-out $(wildcard $(DIRS_USED)),$(DIRS_USED))

$(MEX_EXE_FILES): $(filter $(MEX_EXE_DIR), $(DIRS_NEEDED))
$(MEX_LIB_FILES): $(filter $(MEX_LIB_DIR), $(DIRS_NEEDED))
$(MEX_OBJ_FILES) $(MEX_PREPROC_FILES) $(MEX_ODPN_FILES): \
  $(filter $(MEX_OBJ_DIR)%, $(DIRS_NEEDED))
$(TST_EXE_FILES): $(filter $(TST_EXE_DIR), $(DIRS_NEEDED))
$(TST_LIB_FILES): $(filter $(TST_LIB_DIR), $(DIRS_NEEDED))
$(TST_OBJ_FILES) $(TST_PREPROC_FILES) $(TST_ODPN_FILES): \
  $(filter $(TST_OBJ_DIR)%, $(DIRS_NEEDED))

$(sort $(DIRS_USED)):
	mkdir -p $@

# Make sure that dependency files are built first
ifneq ($(DO_M_DPND)$(DO_T_DPND),YESYES)

all exe obj lib preproc dpnd:
	$(MAKE) GPU=$(GPU) 'DO_M_DPND=YES' 'DO_T_DPND=YES' $@

ifneq ($(MATLAB_PATH),)
all exe obj lib preproc dpnd: preproc-opts-mex
endif

else

all: all-mex all-tst
exe: exe-mex exe-tst
obj: obj-mex obj-tst
lib: lib-mex lib-tst
dpnd: dpnd-mex dpnd-tst
preproc: preproc-mex preproc-tst

endif

ifneq ($(DO_M_DPND),YES)
mex all-mex exe-mex preproc-mex obj-mex lib-mex dpnd-mex \
  $(MEX_PREPROC_FILES) $(MEX_ODPN_FILES) $(MEX_EXE_FILES) \
  $(MEX_LIB_FILES) $(MEX_OBJ_FILES): preproc-opts-mex
	$(MAKE) GPU=$(GPU) 'DO_M_DPND=YES'  $@

else

include $(MEX_ODPN_FILES)

all-mex mex: exe-mex preproc-mex obj-mex lib-mex dirs-mex
exe-mex: $(MEX_EXE_FILES)
preproc-mex: $(MEX_PREPROC_FILES)
obj-mex: $(MEX_OBJ_FILES)
lib-mex: $(MEX_LIB_FILES)
dpnd-mex: $(MEX_ODPN_FILES)

endif

ifneq ($(DO_T_DPND),YES)

tst all-tst exe-tst preproc-tst obj-tst lib-tst data dpnd-tst \
  $(TST_PREPROC_FILES) $(TST_ODPN_FILES) $(TST_EXE_FILES) \
  $(TST_LIB_FILES) $(TST_OBJ_FILES):
	$(MAKE) GPU=$(GPU) 'DO_T_DPND=YES'  $@

else

include $(TST_ODPN_FILES)

all-tst tst: exe-tst preproc-tst obj-tst lib-tst dirs-tst data
exe-tst: $(TST_EXE_FILES)
preproc-tst: $(TST_PREPROC_FILES)
obj-tst: $(TST_OBJ_FILES)
lib-tst: $(TST_LIB_FILES)

data: do_data

endif

dirs: dirs-mex dirs-tst
dirs-tst: $(filter-out $(wildcard $(TST_DIRS_USED)),$(TST_DIRS_USED))
dirs-mex: $(filter-out $(wildcard $(MEX_DIRS_USED)),$(MEX_DIRS_USED))

dpnd-tst: $(TST_ODPN_FILES)

mex: all-mex
tst: all-tst

ifeq ($(DO_M_DPND),YES)

# Rules for MEX executables generation
$(MEX_C_EXE_FILES): $(MEX_EXE_DIR)/%$(MEX_EXT): $(MEX_OBJ_DIR)/%$(MEX_O_EXT)
	@$(MEX_CC_EXE_CMD)
$(MEX_CPP_EXE_FILES): $(MEX_EXE_DIR)/%$(MEX_EXT): $(MEX_OBJ_DIR)/%$(MEX_O_EXT)
	@$(MEX_CPP_EXE_CMD)
$(MEX_CUDA_EXE_FILES): $(MEX_EXE_DIR)/%$(MEX_EXT): $(MEX_OBJ_DIR)/%$(MEX_O_EXT)
	@$(MEX_CUDA_EXE_CMD)

# Rules for MEX library generation
$(MEX_LIB_FILES):
	@$(MEX_LIB_CMD)

# Rules for MEX object files
$(MEX_C_OBJ_FILES): $(MEX_OBJ_DIR)/%$(MEX_O_EXT): %.c
	@$(MEX_CC_OBJ_CMD)
$(MEX_CPP_OBJ_FILES): $(MEX_OBJ_DIR)/%$(MEX_O_EXT): %.cc
	@$(MEX_CPP_OBJ_CMD)
$(MEX_CUDA_OBJ_FILES): $(MEX_OBJ_DIR)/%$(MEX_O_EXT): %.cu
	@$(MEX_CUDA_OBJ_CMD)

# Rules for MEX preprocessed files
$(MEX_C_PREPROC_FILES): $(MEX_OBJ_DIR)/%.i: %.c
	@$(MEX_CC_PREPROC_CMD)
$(MEX_CPP_PREPROC_FILES): $(MEX_OBJ_DIR)/%.i: %.cc
	@$(MEX_CPP_PREPROC_CMD)
$(MEX_CUDA_PREPROC_FILES): $(MEX_OBJ_DIR)/%.i: %.cu
	@$(MEX_CUDA_PREPROC_CMD)

# Rules for object dependency files
$(MEX_C_ODPN_FILES): $(MEX_OBJ_DIR)/%.odpn: $(MEX_OBJ_DIR)/%.i
	@$(MEX_CC_OBJ_DPND_CMD)
$(MEX_CPP_ODPN_FILES): $(MEX_OBJ_DIR)/%.odpn: $(MEX_OBJ_DIR)/%.i
	@$(MEX_CPP_OBJ_DPND_CMD)
$(MEX_CUDA_ODPN_FILES): $(MEX_OBJ_DIR)/%.odpn: $(MEX_OBJ_DIR)/%.i
	@$(MEX_CUDA_OBJ_DPND_CMD)

endif

ifeq ($(DO_T_DPND),YES)

# Rules for TST executables generation
$(TST_C_EXE_FILES): $(TST_EXE_DIR)/%$(EXE_EXT): $(TST_OBJ_DIR)/%$(OBJ_EXT)
	@$(CC_EXE_CMD)
$(TST_CPP_EXE_FILES): $(TST_EXE_DIR)/%$(EXE_EXT): $(TST_OBJ_DIR)/%$(OBJ_EXT)
	@$(CPP_EXE_CMD)
$(TST_CUDA_EXE_FILES): $(TST_EXE_DIR)/%$(EXE_EXT): $(TST_OBJ_DIR)/%$(OBJ_EXT)
	@$(CUDA_EXE_CMD)

# Rules for TST library generation
$(TST_LIB_FILES):
	@$(TST_LIB_CMD)

# Rules for TST object files
$(TST_C_OBJ_FILES): $(TST_OBJ_DIR)/%$(OBJ_EXT): %.c
	@$(TST_CC_OBJ_CMD)
$(TST_CPP_OBJ_FILES): $(TST_OBJ_DIR)/%$(OBJ_EXT): %.cc
	@$(TST_CPP_OBJ_CMD)
$(TST_CUDA_OBJ_FILES): $(TST_OBJ_DIR)/%$(OBJ_EXT): %.cu
	@	$(TST_CUDA_OBJ_CMD)

# Rules for TST preprocessed files
$(TST_C_PREPROC_FILES): $(TST_OBJ_DIR)/%.i: %.c
	@$(CC_PREPROC_CMD)
$(TST_CPP_PREPROC_FILES): $(TST_OBJ_DIR)/%.i: %.cc
	@$(CPP_PREPROC_CMD)
$(TST_CUDA_PREPROC_FILES): $(TST_OBJ_DIR)/%.i: %.cu
	@$(CUDA_PREPROC_CMD)

# Rules for object dependency files
$(TST_C_ODPN_FILES): $(TST_OBJ_DIR)/%.odpn: $(TST_OBJ_DIR)/%.i
	@$(CC_OBJ_DPND_CMD)
$(TST_CPP_ODPN_FILES): $(TST_OBJ_DIR)/%.odpn: $(TST_OBJ_DIR)/%.i
	@$(CPP_OBJ_DPND_CMD)
$(TST_CUDA_ODPN_FILES): $(TST_OBJ_DIR)/%.odpn: $(TST_OBJ_DIR)/%.i
	@$(CUDA_OBJ_DPND_CMD)

endif

preproc-opts-mex: $(PREPROC_OPTS_MEX_FILES)

del-mex: del-all-mex
del-tst: del-all-tst
clean: del-all
	@echo ' **** clean done ****'

del-all: del-all-mex del-all-tst del-data
	@echo ' **** del-all done ****'
del-all-mex: del-exe-mex del-preproc-mex del-obj-mex del-dpnd-mex del-lib-mex del-preproc-opts-mex
	@echo ' **** del-all-mex done ****'
del-all-tst: del-exe-tst del-preproc-tst del-obj-tst del-dpnd-tst del-lib-tst
	@echo ' **** del-all-tst done ****'

del-dirs: del-dirs-mex del-dirs-tst
del-dirs-mex:
	rm -rf $(MEX_DIRS_USED)
del-dirs-tst:
	rm -rf $(TST_DIRS_USED)

del-exe: del-exe-mex del-exe-tst
del-exe-mex:
	@echo "Deleting MEX exectuable files"
	@rm -f $(MEX_EXE_FILES)
del-exe-tst:
	@echo "Deleting TST executable files"
	@rm -f $(TST_EXE_FILES)

del-preproc: del-preproc-mex del-preproc-tst
del-preproc-mex:
	@echo "Deleting MEX preproc files"
	@rm -f $(MEX_PREPROC_FILES)
del-preproc-opts-mex:
	@echo "Deleting MEX preproc-opts files"
	@rm -f $(PREPROC_OPTS_MEX_FILES)
del-preproc-tst:
	@echo "Deleting TST preproc files"
	@rm -f $(TST_PREPROC_FILES)

del-dpnd: del-dpnd-mex del-dpnd-tst
del-dpnd-mex:
	@echo "Deleting MEX dependencies files"
	@rm -f $(MEX_ODPN_FILES) $(MEX_OBJ_DIR)/*.odpn $(MEX_OBJ_DIR)/*.odpn- \
              $(MEX_OBJ_DIR)/*.odpn.err
del-dpnd-tst:
	@echo "Deleting TST dependencies files"
	@rm -f $(TST_ODPN_FILES) $(TST_OBJ_DIR)/*.odpn $(TST_OBJ_DIR)/*.odpn- \
              $(TST_OBJ_DIR)/*.odpn.err

del-data:

del-obj: del-obj-mex del-obj-tst
del-obj-mex:
	@echo "Deleting MEX object files"
	@rm -f $(MEX_OBJ_FILES)
del-obj-tst:
	@echo "Deleting TST object files"
	@rm -f $(TST_OBJ_FILES)

del-lib: del-lib-mex del-lib-tst
del-lib-mex:
	@echo "Deleting MEX libraries files"
	@rm -f $(MEX_LIB_FILES)
del-lib-tst:
	@echo "Deleting TST libraries files"
	@rm -f $(TST_LIB_FILES)

clobber: clobber-mex clobber-tst
clobber-tst: del-exe-tst del-lib-tst
	rm -rf $(TST_OBJ_DIR)
clobber-mex: del-exe-mex del-lib-mex
	rm -rf $(MEX_OBJ_DIR)

do_data:

