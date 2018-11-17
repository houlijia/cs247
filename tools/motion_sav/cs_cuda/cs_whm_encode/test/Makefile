all : cs_iwhm_test

cs_iwhm_test : cs_iwhm_test.cu ../../cs_misc/cs_dbg.o ../../cs_misc/cs_helper.o \
	../../cs_misc/cs_dct.o ../cs_whm_encode_b.o 
	nvcc --compiler-bindir $(CUDA_CC) -I.. -I../../cs_misc cs_iwhm_test.cu \
		../../cs_misc/cs_dbg.o \
		../../cs_misc/cs_helper.o \
		../../cs_misc/cs_dct.o \
		../cs_whm_encode_b.o \
		-lm -lgomp -o cs_iwhm_test

clean:
	rm -f cs_iwhm_test
