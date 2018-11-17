
NVCC=nvcc
NVCC_CMPL_ONLY:=-dc

# OS Specific settings

ifeq ($(TARGET_OS),Linux)
NVCC_OPTS+=

else
ifneq ($(findstring CYGWIN,$(TARGET_OS)),)
NVCC_OPTS+=-idp /cygwin/
else
NVCC_OPTS+=
endif
endif  				# ifeq ($(TARGET_OS),Linux)

ifeq ($(CUDA_CC),)
$(error missing environment variable: CUDA_CC)
endif

ifeq ($(CUDA_CPLUSPLUS),)
$(error missing environment variable: CUDA_CPLUSPLUS)
endif

define CUDA_OBJ_DPND_CMD
$(call DPND_OBJ_TMPLT,.odpn, %.i %$(OBJ_EXT)) 
endef

define CUDA_PREPROC_CMD
$(call PREPROC_TMPLT, $(NVCC) -E $(NVCC_OPTS), %.c %.cc %.cu,) 
endef

define TST_CUDA_OBJ_CMD
@echo "$(patsubst $(CURDIR)/%,%,$<) --> $(patsubst $(ROOT_DIR)/%,%,$@)"
$(NVCC) $(NVCC_CMPL_ONLY) $(NVCC_OPTS) $< -o $@
endef

define CUDA_EXE_CMD
@echo '==>' $(patsubst $(ROOT_DIR)/%,%,$@)
$(NVCC) $(NVCC_OPTS) $($*_OPTS) -o $@ $(filter $(SRC_FLTR) %$(OBJ_EXT), $^) \
  $(filter %$(LIB_EXT), $^) $(TST_CUDA_SYS_LIBS)
endef

# From here on, it is relevant only to Mex.
ifneq ($(MATLAB_PATH),)

# Add CUDA path to MEX include path
ifeq ($(filter %/gpu,$(MEX_INCLUDE)),)

MEX_INCLUDE+= -I"$(MATLAB_PATH)/"toolbox/distcomp/gpu/extern/include
endif

PREPROC_OPTS_MEX_FILES+= $(MEX_OBJ_DIR)/mex_cuda_opts.out $(MEX_OBJ_DIR)/mex_cuda_preproc.mak
$(MEX_OBJ_DIR)/mex_cuda_preproc.mak: $(MEX_OBJ_DIR)/mex_cuda_opts.out
	@echo '==>' $(patsubst $(ROOT_DIR)/%,%,$@)
	@rm -f $@
ifeq ($(TARGET_OS),Linux)
	@echo 'MX_CUDA_PREPROC:= $(shell sed -n \
		'/^[\t ]*CMDLINE1[\t ]*:/{s/^[\t ]*CMDLINE1[\t ]*:[\t ]*//;s/[^\t ]*_dummy.cu//;s/-o[\t ]*[^\t ]*_dummy$(MEX_O_EXT)//;s|[\t ][-/]c[\ ]| |;s|[\t ]-DNDEBUG[^\t ]*||g;s|$$| -E $$(MEX_PRPRC_OPTS) |p;q}' \
		$(MEX_OBJ_DIR)/mex_cuda_opts.out)' >> $@
else				# ifeq ($(TARGET_OS),Linux)
	@echo 'MX_CUDA_OPTS_FILE:=$(shell sed -n \
		'/^[\t ]*Options[\t ]*file[\t ]*:/{s/^[\t ]*Options[\t ]*file[\t ]*:[\t ]*//;s/[\t ]*$$//;s/[\t ]*$$//;s|\\|/|g;p;q}'\
                $(MEX_OBJ_DIR)/mex_cuda_opts.out)' >> $@
endif				# ifeq ($(TARGET_OS),Linux)

$(MEX_OBJ_DIR)/mex_cuda_opts.out: $(filter-out $(wildcard $(MEX_OBJ_DIR)),$(MEX_OBJ_DIR))
	@echo '==>' $(patsubst $(ROOT_DIR)/%,%,$@)
	@cd $(@D); echo "" > _dummy.cu; \
	  echo mexcuda -c -v -n $(subst ',\',$(MEX_OPTS)) $(MEX_INCLUDE)\
	  _dummy.cu > mex_cuda_opts_bld.m
	@cd $(@D);  matlab -nodisplay -nojvm -r  'diary  $(@F); mex_cuda_opts_bld; exit;' > /dev/null && \
          rm -f mex_cuda_opts_bld.m _dummy.cu

ifeq ($(DO_M_DPND),YES)

include $(MEX_OBJ_DIR)/mex_cuda_preproc.mak

MEX_CUDA_PREPROC:=$(subst @@@,\#,$(MX_CUDA_PREPROC))

endif

define MEX_CUDA_OBJ_DPND_CMD
$(call DPND_OBJ_TMPLT,.odpn, %.i %$(MEX_O_EXT))
endef

define MEX_CUDA_PREPROC_CMD
$(call PREPROC_TMPLT, $(MEX_CUDA_PREPROC), %.c %.cc %.cu) 
endef

ifneq ($(findstring CYGWIN,$(TARGET_OS)),) # Cygnwin case. mex output options does not work well

define MEX_CUDA_OBJ_CMD
@echo "$(<F) --> $(patsubst $(ROOT_DIR)/%,%,$@)"
rm -f $*_bld.m
echo mexcuda -c $(subst ',\',$(MEX_OPTS)) $(MEX_INCLUDE) $< > $*_bld.m
matlab -nodisplay -nojvm -r "$*_bld; exit;" && rm $*_bld.m && mv $(@D) $@
endef

define MEX_CUDA_EXE_CMD
@echo '==>' $(patsubst $(ROOT_DIR)/%,%,$@)
rm -f $*_bld.m
echo mexcuda $(subst ',\',$(MEX_OPTS)) -output $@ \
  $(filter $(SRC_FLTR) %$(MEX_O_EXT), $^ ) \
  $(filter %$(MEX_L_EXT), $^) $(MEX_LIBS) > $*_bld.m
matlab -nodisplay -nojvm -r "$*_bld; exit;" && rm $*_bld.m && mv $(@D) $@
endef

else  # Linux case 

define MEX_CUDA_OBJ_CMD
@echo "$(<F) --> $(patsubst $(ROOT_DIR)/%,%,$@)"
rm -f $*_bld.m
echo mexcuda -c $(subst ',\',$(MEX_OPTS)) $(MEX_INCLUDE) -outdir $(@D) $< > $*_bld.m
matlab -nodisplay -nojvm -r "$*_bld; exit;" && rm $*_bld.m
endef

define MEX_CUDA_EXE_CMD
@echo '==>' $(patsubst $(ROOT_DIR)/%,%,$@)
rm -f $*_bld.m
echo mexcuda $(subst ',\',$(MEX_OPTS)) -outdir $(@D) -output $@ \
  $(filter $(SRC_FLTR) %$(MEX_O_EXT), $^ ) \
  $(filter %$(MEX_L_EXT), $^) $(MEX_LIBS) > $*_bld.m
matlab -nodisplay -nojvm -r "$*_bld; exit;" && rm $*_bld.m
endef

endif				#ifneq ($(findstring CYGWIN,$(TARGET_OS)),)

endif				# ifneq ($(MATLAB_PATH),)
