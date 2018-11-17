all : cs_whm_encode.o cs_whm_encode_b.o

cs_whm_encode_b.o : cs_whm_encode_b.cu cs_whm_encode_b.h ../cs_misc/cs_dbg.h ../cs_misc/cs_helper.h
	nvcc --compiler-bindir $(CUDA_CC) -I../cs_misc -c cs_whm_encode_b.cu

cs_whm_encode.o : cs_whm_encode.cu cs_whm_encode.h
	nvcc --compiler-bindir $(CUDA_CC) -c cs_whm_encode.cu

clean:
	rm -f cs_whm_encode.o
	rm -f cs_whm_encode_b.o
