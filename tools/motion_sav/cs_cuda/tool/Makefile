all : capture parser1 parser prank dataskip datar2c dataa2b datab2a idx whm2 whm whmt \
	datagen datagenb cuda_grid_test cuda_syn_test cuda_grid_blk_test serial_wht find_thread_blk \
	cuda_syn2_test serial_wht2 uyvy2yuv422p serial_wht3 reshape

CS_MISC=../cs_misc
CS_WHM_ENCODE=../cs_whm_encode
RNDSTRMC=../..

NVCC_OPTS=-arch=compute_35

capture : capture.cu $(CS_MISC)/cs_dbg.o $(CS_MISC)/cs_helper.o \
  $(CS_MISC)/cs_block.o $(CS_MISC)/cs_config.o $(CS_MISC)/cs_video_io.o  \
  $(CS_MISC)/cs_webcam.o 

	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I$(CS_WHM_ENCODE) -I$(CS_MISC) -I$(RNDSTRMC)/RndStrmC \
        -I$(RNDSTRMC)/RndStrmC/cgen/lib capture.cu \
        $(CS_MISC)/cs_dbg.o \
        $(CS_MISC)/cs_helper.o \
        $(CS_MISC)/cs_config.o \
        $(CS_MISC)/cs_video_io.o \
        $(CS_MISC)/cs_webcam.o \
		-L/usr/local/lib \
        -lopencv_calib3d \
        -lopencv_contrib \
        -lopencv_core \
        -lopencv_features2d \
        -lopencv_flann \
        -lopencv_gpu \
        -lopencv_highgui \
        -lopencv_imgproc \
        -lopencv_legacy \
        -lopencv_ml \
        -lopencv_nonfree \
        -lopencv_objdetect \
        -lopencv_photo \
        -lopencv_stitching \
        -lopencv_superres \
        -lopencv_ts \
        -lopencv_video \
        -lopencv_videostab \
        -ljson-c -lm -lgomp -o capture

uyvy2yuv422p : uyvy2yuv422p.c 
	cc uyvy2yuv422p.c -g -o uyvy2yuv422p

dataskip : dataskip.c 
	cc dataskip.c -g -o dataskip

dataa2b : dataa2b.c 
	cc dataa2b.c -g -o dataa2b

datar2c : datar2c.c 
	cc datar2c.c -g -o datar2c

datab2a : datab2a.c 
	cc datab2a.c -g -o datab2a

datagenb : datagenb.c 
	cc datagenb.c -g -o datagenb

find_thread_blk : find_thread_blk.c 
	cc find_thread_blk.c -g -lm -o find_thread_blk

datagen : datagen.c 
	cc datagen.c -g -o datagen

reshape : reshape.c 
	cc reshape.c -g -lm -o reshape

serial_wht3 : serial_wht3.c 
	cc serial_wht3.c -g -lm -o serial_wht3

serial_wht2 : serial_wht2.c 
	cc serial_wht2.c -g -lm -o serial_wht2

serial_wht : serial_wht.c 
	cc serial_wht.c -g -lm -o serial_wht

whmt : whmt.c 
	cc whmt.c -g -lm -o whmt

idx : idx.c 
	cc idx.c -g -lm -o idx

whm : whm.c 
	cc whm.c -g -lm -o whm

whm2 : whm2.c 
	cc whm2.c -g -lm -o whm2

prank : prank.c 
	cc prank.c -g -lm -o prank

parser1 : parser1.c 
	cc parser1.c -g -lm -o parser1

parser : parser.c 
	cc parser.c -g -lm -o parser

cuda_syn2_test : cuda_syn2_test.cu $(CS_MISC)/cs_dbg.o $(CS_MISC)/cs_helper.o 
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I$(CS_MISC) cuda_syn2_test.cu \
        $(CS_MISC)/cs_dbg.o \
        $(CS_MISC)/cs_helper.o \
        -lgomp -o cuda_syn2_test

cuda_grid_blk_test : cuda_grid_blk_test.cu $(CS_MISC)/cs_dbg.o $(CS_MISC)/cs_helper.o 
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I$(CS_MISC) cuda_grid_blk_test.cu \
        $(CS_MISC)/cs_dbg.o \
        $(CS_MISC)/cs_helper.o \
        -lgomp -o cuda_grid_blk_test

cuda_grid_test : cuda_grid_test.cu $(CS_MISC)/cs_dbg.o $(CS_MISC)/cs_helper.o 
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I$(CS_MISC) cuda_grid_test.cu \
        $(CS_MISC)/cs_dbg.o \
        $(CS_MISC)/cs_helper.o \
        -lgomp -o cuda_grid_test

cuda_syn_test : cuda_syn_test.cu $(CS_MISC)/cs_dbg.o $(CS_MISC)/cs_helper.o 
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I$(CS_MISC) cuda_syn_test.cu \
        $(CS_MISC)/cs_dbg.o \
        $(CS_MISC)/cs_helper.o \
        -lgomp -o cuda_syn_test

clean :
	rm whm
	rm whmt
	rm datagen
	rm datagenb
	rm datab2a
	rm dataa2b
	rm datar2c
	rm idx
	rm whm2
	rm prank
	rm parser
	rm parser1
	rm cuda_grid_test
	rm cuda_syn_test
	rm find_thread_blk
	rm cuda_syn2_test
	rm uyvy2yuv422p
	rm serial_wht
	rm serial_wht2
	rm serial_wht3
