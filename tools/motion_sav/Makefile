# \file
#
# This is a makefile which invokes all the makefiles in sub directories

export MEXCC:=
export MATLAB_PATH:=

ifeq ($(ROOT_DIR),)
ROOT_DIR:=$(realpath $(patsubst %/,%,$(shell cd ..; pwd)))
endif

include $(ROOT_DIR)/util/basic_defs.mak

DIRS:= $(patsubst %,cs_cuda/%, cs_misc cs_whm_encode\
  $(patsubst %,proj/%, acq_ana.file camera_demo camera_demo/tcp_socket \
                       obj_recog obj_recog/Nimresize))
TST_DIRS:=$(DIRS)
OBJ_DIRS:=$(patsubst %,cs_cuda/%, cs_misc cs_whm_encode \
  $(patsubst %,proj/%, camera_demo/tcp_socket acq_ana.file camera_demo obj_recog))
LIB_DIRS:=$(patsubst %,cs_cuda/%, cs_misc cs_whm_encode \
  proj/camera_demo proj/camera_demo/tcp_socket)
DPND_DIRS:=$(patsubst %,cs_cuda/%, cs_misc cs_whm_encode \
  $(patsubst %,proj/%, camera_demo/tcp_socket acq_ana.file camera_demo obj_recog))
EXE_DIRS:=$(patsubst %,cs_cuda/%, cs_misc \
  $(patsubst %,proj/%, acq_ana.file camera_demo obj_recog))

BLD_TRGTS:= all exe preproc obj lib dpnd dirs
TST_BLD_TRGTS:=$(patsubst %,%-tst,$(BLD_TRGTS)) tst
MEX_BLD_TRGTS:=$(patsubst %,%-mex,$(BLD_TRGTS)) mex
DEL_TRGTS:=$(patsubst %,del-%,$(BLD_TRGTS))
TST_DEL_TRGTS:=$(patsubst %,%-tst,$(DEL_TRGTS))
MEX_DEL_TRGTS:=$(patsubst %,%-mex,$(DEL_TRGTS))

TRGTS:=$(BLD_TRGTS) $(DEL_TRGTS) clean

DIR_TRGTS:=$(foreach T, $(TRGTS), $(patsubst %,$T-+-%,$(DIRS)))

.PHONY: $(TRGTS) $(DIR_TRGTS) clobber

all:             $(patsubst %, all-+-%,             $(DIRS))
exe:             $(patsubst %, exe-+-%,             $(EXE_DIRS)) 
preproc:         $(patsubst %, preproc-+-%,         $(OBJ_DIRS))
obj:             $(patsubst %, obj-+-%,             $(OBJ_DIRS))
lib:             $(patsubst %, lib-+-%,             $(DPND_DIRS))
dpnd:            $(patsubst %, dpnd-+-%,            $(DPND_DIRS)) 
dirs:            $(patsubst %, dirs-+-%,            $(OBJ_DIRS) $(EXE_DIRS) $(LIB_DIRS))
del-all del-tst: clean
clean:           $(patsubst %, clean-+-%,           $(DIRS))
del-exe:         $(patsubst %, del-exe-+-%,         $(EXE_DIRS))
del-preproc:     $(patsubst %, del-preproc-+-%,     $(OBJ_DIRS))
del-obj:         $(patsubst %, del-obj-+-%,         $(OBJ_DIRS))
del-lib:         $(patsubst %, del-lib-+-%,         $(LIB_DIRS))
del-dpnd:        $(patsubst %, del-dpnd-+-%,        $(DPND_DIRS))
del-dirs:        $(patsubst %, del-dirs-+-%,        $(OBJ_DIRS) $(LIB_DIRS) $(DPND_DIRS))

tst: all
del-tst: del-all

$(MEX_BLD_TRGTS) $(MEX_DEL_TRGTS): ;

$(patsubst %, all-+-%,             $(DIRS)): lib
$(patsubst %, exe-+-%,             $(EXE_DIRS)): lib
$(patsubst %, all-tst-+-%,         $(DIRS)): all-tst-+-% :     all-+-% ;
$(patsubst %, exe-tst-+-%,         $(DIRS)): exe-tst-+-% :     exe-+-% ;
$(patsubst %, lib-tst-+-%,         $(DIRS)): lib-tst-+-% :     lib-+-% ;
$(patsubst %, obj-tst-+-%,         $(DIRS)): obj-tst-+-% :     obj-+-% ;
$(patsubst %, preproc-tst-+-%,     $(DIRS)): preproc-tst-+-% : preproc-+-% ;
$(patsubst %, dpnd-tst-+-%,        $(DIRS)): dpnd-tst-+-% :    dpnd-+-% ;
$(patsubst %, dirs-tst-+-%,        $(DIRS)): dirs-tst-+-% :    dirs-+-% ;
$(patsubst %, tst-+-%,             $(DIRS)): tst-+-% :         all-+-% ;
$(patsubst %, del-tst-+-%,         $(DIRS)): del-tst-+-% :     del-all-+-% ;
$(patsubst %, del-all-tst-+-%,     $(DIRS)): del-all-tst-+-% : del-all-+-% ;
$(patsubst %, del-exe-tst-+-%,     $(DIRS)): del-exe-tst-+-% : del-exe-+-% ;
$(patsubst %, del-lib-tst-+-%,     $(DIRS)): del-lib-tst-+-% : del-lib-+-% ;
$(patsubst %, del-obj-tst-+-%,     $(DIRS)): del-obj-tst-+-% : del-obj-+-% ;
$(patsubst %, del-preproc-tst-+-%, $(DIRS)): del-preproc-tst-+-% : del-preproc-+-% ;
$(patsubst %, del-dpnd-tst-+-%,    $(DIRS)): del-dpnd-tst-+-% : del-dpnd-+-% ;
$(patsubst %, del-dirs-tst-+-%,    $(DIRS)): del-dirs-tst-+-% :  del-dirs-+-% ;

clobber: del-exe del-lib
	rm -rf $(TST_DIR)/obj/motion_sav

$(sort $(DIR_TRGTS) $(foreach T, all clean, $(patsubst %, $T-+-%, $(DIRS)))):
	@echo "Building '$(subst -+-,' in ,$@)"
	$(MAKE) $(OVER) ROOT_DIR=$(ROOT_DIR) CFG=$(CFG) $(subst -+-, -C ,$@)
