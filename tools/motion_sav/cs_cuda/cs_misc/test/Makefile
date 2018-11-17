# include path
INCLUDE_DIRS:=.. ../../../../RndStrmC ../../../../RndStrmC/cgen/lib
INCLUDE_OPT:=$(patsubst %, -I%,$(INCLUDE_DIRS))

include $(ROOT_DIR)/util/defs.mak

RND_LIB:=$(TST_LIB_DIR)/libcgenrnd.a

all : cs_matrix_test cs_dct_test cs_perm_generic_test cs_helper_test cs_dbg_test cs_complgrngn_test\
		cs_vector_test cs_random_test cs_domultivec_test cs_sparser_test cs_compGrad_x_test \
		cs_buffer_test cs_decode_parser_test cs_quantize_test cs_perm_mlseq_test cs_syncthread_test \
		cs_mean_sd_test cs_do_quant


NVCC_OPTS=-arch=compute_35

cs_random_test : cs_random_test.cu ../cs_dbg.o ../cs_helper.o ../cs_random.o
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) $(INCLUDE_OPT)\
		cs_random_test.cu \
		../cs_dbg.o \
		../cs_helper.o \
		../cs_random.o \
		$(RND_LIB) \
		-lm -lgomp -o cs_random_test

cs_do_quant : cs_do_quant.cu ../cs_mean_sd.o ../cs_dbg.o ../cs_helper.o ../cs_quantize.o \
	../cs_motion_detect_v4.o ../cs_mean_sd.o ../cs_copy_box.o
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) $(INCLUDE_OPT)\
		cs_do_quant.cu \
		../cs_dbg.o \
		../cs_helper.o \
		../cs_mean_sd.o \
		../cs_copy_box.o \
		../cs_motion_detect_v4.o \
		../cs_quantize.o \
		-lm -lgomp -o cs_do_quant

cs_quantize_test : cs_quantize_test.cu ../cs_matrix.o ../cs_dbg.o ../cs_helper.o ../cs_quantize.o \
	../cs_motion_detect_v4.o
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) $(INCLUDE_OPT)\
		cs_quantize_test.cu \
		../cs_dbg.o \
		../cs_helper.o \
		../cs_matrix.o \
		../cs_decode_parser.o \
		../cs_motion_detect_v4.o \
		../cs_quantize.o \
		-lm -lgomp -o cs_quantize_test

cs_mean_sd_test : cs_mean_sd_test.cu ../cs_dbg.o ../cs_helper.o ../cs_decode_parser.o \
		../cs_copy_box.o ../cs_mean_sd.o ../cs_motion_detect_v4.o
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) $(INCLUDE_OPT)\
		cs_mean_sd_test.cu \
		../cs_dbg.o \
		../cs_helper.o \
		../cs_mean_sd.o \
		../cs_motion_detect_v4.o \
		../cs_copy_box.o \
		../cs_decode_parser.o \
		-lm -lgomp -o cs_mean_sd_test

cs_decode_parser_test : cs_decode_parser_test.cu ../cs_dbg.o ../cs_helper.o ../cs_decode_parser.o
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) $(INCLUDE_OPT)\
		cs_decode_parser_test.cu \
		../cs_dbg.o \
		../cs_helper.o \
		../cs_decode_parser.o \
		-lm -lgomp -o cs_decode_parser_test

cs_domultivec_test : cs_domultivec_test.cu ../cs_dbg.o ../cs_helper.o ../cs_random.o \
	   ../cs_domultivec.o ../cs_perm_generic.o ../../cs_whm_encode/cs_whm_encode_b.o
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) $(INCLUDE_OPT)\
		cs_domultivec_test.cu \
		../../cs_whm_encode/cs_whm_encode_b.o \
		../cs_perm_generic.o \
		../cs_dbg.o \
		../cs_helper.o \
		../cs_random.o \
		../cs_domultivec.o \
		../../../RndStrmC/libcsgenlib.a \
		-lm -lgomp -o cs_domultivec_test

cs_sparser_test : cs_sparser_test.cu ../cs_dbg.o ../cs_helper.o ../cs_sparser.o ../cs_dct.o \
	../cs_matrix.o
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) $(INCLUDE_OPT)\
		cs_sparser_test.cu \
		../cs_dbg.o \
		../cs_helper.o \
		../cs_sparser.o \
		../cs_dct.o \
		../cs_matrix.o \
		-lm -lgomp -o cs_sparser_test

cs_compGrad_x_test : cs_compGrad_x_test.cu ../cs_dbg.o ../cs_helper.o ../cs_sparser.o \
	../cs_dct.o ../cs_vector.o ../cs_perm_generic.o ../cs_matrix.o ../cs_compGrad_x.o \
	../../cs_whm_encode/cs_whm_encode_b.o ../cs_buffer.o ../cs_random.o
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) $(INCLUDE_OPT) \
		cs_compGrad_x_test.cu \
		../cs_dbg.o \
		../cs_helper.o \
		../cs_sparser.o \
		../cs_vector.o \
		../cs_perm_generic.o \
		../../cs_whm_encode/cs_whm_encode_b.o \
		../cs_dct.o \
		../../../RndStrmC/libcsgenlib.a \
		../cs_compGrad_x.o \
		../cs_matrix.o \
		../cs_buffer.o \
		../cs_random.o \
		-lm -lgomp -o cs_compGrad_x_test

cs_buffer_test : cs_buffer_test.cu ../cs_dbg.o ../cs_helper.o ../cs_buffer.o
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I.. cs_buffer_test.cu \
		../cs_dbg.o \
		../cs_buffer.o \
		../cs_helper.o \
		-lm -lgomp -o cs_buffer_test

cs_vector_test : cs_vector_test.cu ../cs_dbg.o ../cs_helper.o ../cs_vector.o
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I.. cs_vector_test.cu \
		../cs_dbg.o \
		../cs_helper.o \
		../cs_vector.o \
		-lm -lgomp -o cs_vector_test

cs_complgrngn_test : cs_complgrngn_test.cu ../cs_dbg.o ../cs_helper.o ../cs_complgrngn.o
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) $(INCLUDE_OPT) \
		cs_complgrngn_test.cu \
		../cs_dbg.o \
		../cs_helper.o \
		../cs_complgrngn.o \
		../cs_matrix.o \
		-lm -lgomp -o cs_complgrngn_test

cs_syncthread_test : cs_syncthread_test.cu ../cs_dbg.o ../cs_helper.o
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I.. cs_syncthread_test.cu \
		../cs_dbg.o \
		../cs_helper.o \
		../cs_matrix.o \
		-lm -lgomp -o cs_syncthread_test

cs_dbg_test : cs_dbg_test.cu ../cs_dbg.o ../cs_helper.o
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I.. cs_dbg_test.cu \
		../cs_dbg.o \
		../cs_helper.o \
		../cs_matrix.o \
		-lm -lgomp -o cs_dbg_test

cs_helper_test : cs_helper_test.cu ../cs_dbg.o ../cs_helper.o
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I.. cs_helper_test.cu \
		../cs_dbg.o \
		../cs_helper.o \
		../cs_matrix.o \
		-lm -lgomp -o cs_helper_test

cs_matrix_test : cs_matrix_test.cu ../cs_dbg.o ../cs_helper.o \
	../cs_matrix.o 
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I.. cs_matrix_test.cu \
		../cs_dbg.o \
		../cs_helper.o \
		../cs_matrix.o \
		-lm -lgomp -o cs_matrix_test

cs_dct_test : cs_dct_test.cu ../cs_dbg.o ../cs_helper.o \
	../cs_dct.o 
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I.. cs_dct_test.cu \
		../cs_dbg.o \
		../cs_helper.o \
		../cs_dct.o \
		-lm -lgomp -o cs_dct_test

cs_perm_generic_test : cs_perm_generic_test.cu ../cs_dbg.o ../cs_helper.o \
	../cs_random.o ../cs_perm_generic.o ../cs_random.o 
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) $(INCLUDE_OPT) \
		cs_perm_generic_test.cu \
		../cs_dbg.o \
		../cs_helper.o \
		../cs_random.o \
		../../cs_whm_encode/cs_whm_encode_b.o \
		../../../RndStrmC/libcsgenlib.a \
		../cs_perm_generic.o \
		-lm -lgomp -o cs_perm_generic_test

cs_perm_mlseq_test : cs_perm_mlseq_test.cu ../cs_dbg.o ../cs_helper.o \
	../cs_random.o ../cs_perm_mlseq.o ../cs_random.o 
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) $(INCLUDE_OPT) \
		cs_perm_mlseq_test.cu \
		../cs_dbg.o \
		../cs_helper.o \
		../cs_random.o \
		../../cs_whm_encode/cs_whm_encode_b.o \
		../../../RndStrmC/libcsgenlib.a \
		../cs_perm_mlseq.o \
		-lm -lgomp -o cs_perm_mlseq_test

clean:
	rm -f cs_matrix_test cs_dct_test cs_perm_generic_test cs_helper_test cs_dbg_test cs_complgrngn_test\
		cs_vector_test cs_random_test cs_domultivec_test cs_sparser_test cs_compGrad_x_test \
		cs_buffer_test cs_do_quant

