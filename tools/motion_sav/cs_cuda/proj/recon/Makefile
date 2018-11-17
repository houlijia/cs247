all : recon

CS_MISC:=\
	../../cs_misc/cs_block.o \
	../../cs_misc/cs_copy_box.o \
	../../cs_misc/cs_edge_detect_v2.o \
	../../cs_misc/cs_matrix.o \
	../../cs_misc/cs_perm_selection.o \
	../../cs_misc/cs_buffer.o \
	../../cs_misc/cs_dbg.o \
	../../cs_misc/cs_expand.o \
	../../cs_misc/cs_motion_detect.o \
	../../cs_misc/cs_random.o \
	../../cs_misc/cs_compGrad_x.o \
	../../cs_misc/cs_dct.o \
	../../cs_misc/cs_header.o \
	../../cs_misc/cs_motion_detect_v2.o \
	../../cs_misc/cs_sparser.o \
	../../cs_misc/cs_complgrngn.o \
	../../cs_misc/cs_decode_parser.o \
	../../cs_misc/cs_helper.o \
	../../cs_misc/cs_motion_report.o \
	../../cs_misc/cs_vector.o \
	../../cs_misc/cs_comp_step.o \
	../../cs_misc/cs_domultivec.o \
	../../cs_misc/cs_interpolate.o \
	../../cs_misc/cs_perm_generic.o \
	../../cs_misc/cs_perm_mlseq.o \
	../../cs_misc/cs_quantize.o \
	../../cs_misc/cs_motion_detect_v4.o \
	../../cs_whm_encode/cs_whm_encode_b.o 

# include path
RNDSTRMC=../../../../RndStrmC
INCLUDE_DIRS:=. ../../cs_misc ../../cs_whm_encode $(RNDSTRMC) $(RNDSTRMC)/cgen/lib
INCLUDE_OPT:=$(patsubst %, -I%,$(INCLUDE_DIRS))

include $(ROOT_DIR)/util/defs.mak

RND_LIB:=$(TST_LIB_DIR)/libcgenrnd.a

recon : recon.cu $(CS_MISC) $(RND_LIB)
	nvcc -arch=compute_35 --compiler-bindir $(CUDA_CC) $(INCLUDE_OPT) \
		$(CS_MISC) $(RND_LIB) -lm -lgomp -o recon

clean :
	rm -f recon
