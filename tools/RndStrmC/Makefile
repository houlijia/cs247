# include path
INCLUDE_DIRS:=. cgen/lib
INCLUDE_OPT:=$(patsubst %, -I%,$(INCLUDE_DIRS))

include ../util/defs.mak

CC_OPTS:=$(subst -O2,-O,$(CC_OPTS)) -D_GNU_SOURCE -ansi -fno-omit-frame-pointer
TST_CC_SYS_LIBS+= -lm
WINDOWS:=$(findstring cmd.exe,$(COMSPEC))

ifeq ($(WINDOWS),) # Not windows

CC_OPTS += -pthread

endif

TST_DIRS_USED+= $(TST_OBJ_DIR)/cgen/lib
ifneq ($(MEXCC),)
MEX_DIRS_USED+= $(MEX_OBJ_DIR)/cgen/lib
endif
DIRS_USED+= $(TST_OBJ_DIR)/cgen/lib $(MEX_OBJ_DIR)/cgen/lib

CGEN_SRCS:=$(patsubst %,cgen/lib/%.c, \
        eml_rand eml_rand_mcg16807_stateful eml_rand_mt19937ar_stateful \
	eml_randn eml_rand_shr3cong_stateful get_rand get_rand_data \
	get_rand_emxAPI get_rand_emxutil get_randi get_rand_initialize \
	get_randn get_randperm get_rand_terminate rand randn \
	randperm rng)

COMMON_SRCS:=RndC_ifc.c RndCState.c

ifneq ($(MEXCC),)

COMMON_MEX_SRCS:=common_RndC_mex.c RndCState_mex.c
MEX_SRCS:=init_RndC_mex.c rand_RndC_mex.c randi_RndC_mex.c randn_RndC_mex.c\
   randperm_RndC_mex.c 
MEX_C_SRCS:=$(CGEN_SRCS) $(CGEN_SRCS) $(COMMON_SRCS) $(COMMON_MEX_SRCS) $(MEX_SRCS)

endif

TST_SRCS:=RndC_test.c

TST_C_SRCS:=$(CGEN_SRCS) $(COMMON_SRCS) $(TST_SRCS)

ifneq ($(MEXCC),)
MEX_COMMON_OBJ:=$(patsubst %.c,$(MEX_OBJ_DIR)/%$(MEX_O_EXT), \
   $(CGEN_SRCS) $(COMMON_SRCS) $(COMMON_MEX_SRCS))
endif

TST_COMMON_OBJ:=$(patsubst %.c,$(TST_OBJ_DIR)/%$(OBJ_EXT), \
  $(CGEN_SRCS) $(COMMON_SRCS))

ifneq ($(MEXCC),)
MEX_CGENRNDLIB:=$(MEX_LIB_DIR)/libcgenrnd$(MEX_L_EXT)
$(MEX_CGENRNDLIB): $(MEX_COMMON_OBJ)
MEX_LIB_FILES:=$(MEX_CGENRNDLIB)
endif				# ifneq ($(MEXCC),)

TST_CGENRNDLIB:=$(TST_LIB_DIR)/libcgenrnd$(LIB_EXT)
$(TST_CGENRNDLIB): $(TST_COMMON_OBJ)
TST_LIB_FILES:=$(TST_CGENRNDLIB)

ifneq ($(MEXCC),)
MEX_C_EXE_FILES:=$(patsubst %.c,$(MEX_EXE_DIR)/%$(MEX_EXT), $(MEX_SRCS))
$(MEX_C_EXE_FILES): $(MEX_CGENRNDLIB)
endif

TST_C_EXE_FILES:=$(patsubst %.c,$(TST_EXE_DIR)/%$(EXE_EXT), $(TST_SRCS))
$(TST_C_EXE_FILES): $(TST_CGENRNDLIB)

include ../util/rules.mak
