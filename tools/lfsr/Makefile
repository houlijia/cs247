# include path
INCLUDE_DIRS:=.
INCLUDE_OPT:=$(patsubst %, -I%,$(INCLUDE_DIRS))

include ../util/defs.mak

LFSR_EXE:=$(TST_EXE_DIR)/gnrt_lfsr$(EXE_EXT)

TST_C_SRCS:= gnrt_lfsr.c comp_lfsr.c parse_opt.c prim_poly_read.c
TST_C_EXE_FILES:=$(LFSR_EXE)

$(LFSR_EXE): $(patsubst %.c,$(TST_OBJ_DIR)/%$(OBJ_EXT), $(TST_C_SRCS))

MLSEQ_DIR:=../../mlseq
DIRS_USED+= $(MLSEQ_DIR)

LFSR_MAX_ORDER:=26

LFSR_WILD:=$(MLSEQ_DIR)/lfsr_*_seq.dat
LFSR_SPEC:=$(subst *,%d,$(LFSR_WILD))
LFSR_TMPLT:=$(subst *,%,$(LFSR_WILD))
LFSR_FILES:=$(patsubst %, $(LFSR_TMPLT), $(shell seq 1 $(LFSR_MAX_ORDER)))
WH_WILD:=$(MLSEQ_DIR)/WH_lfsr_*_indcs.dat
WH_SPEC:=$(subst *,%d,$(WH_WILD))
WH_TMPLT:=$(subst *,%,$(WH_WILD))
WH_FILES:=$(patsubst %, $(WH_TMPLT), $(shell seq 1 $(LFSR_MAX_ORDER)))

ifneq (,$(filter-out $(wildcard $(LFSR_FILES) $(WH_FILES)), $(LFSR_FILES) $(WH_FILES)))

do_data: exe-tst $(filter-out $(wildcard $(MLSEQ_DIR)),$(MLSEQ_DIR))
	$(LFSR_EXE) -o $(LFSR_SPEC) -w $(WH_SPEC) -v1 $(LFSR_MAX_ORDER)

endif

del-data:
	@rm -f $(LFSR_WILD) $(WH_WILD)
	@echo ' **** del-data done ****'

clobber: clobber-data

clobber-data:
	rm -rf $(MLSEQ_DIR)

include ../util/rules.mak
