# include path
INCLUDE_DIRS:=. ../../../RndStrmC ../../../RndStrmC/cgen/lib \
  ../cs_whm_encode /usr/local/include/json /usr/local/include/opencv /usr/local/include/opencv
INCLUDE_OPT:=$(patsubst %, -I%,$(INCLUDE_DIRS))

include $(ROOT_DIR)/util/defs.mak

TST_C_SRCS:=readrl.c
TST_CPP_SRCS:=cs_image.cc
TST_CUDA_SRCS:=$(patsubst %,%.cu, \
        pheader cs_perm_mlseq cs_interpolate cs_expand \
	cs_header cs_helper cs_dbg cs_block cs_perm_selection \
	cs_edge_detect cs_motion_detect cs_motion_detect_v2 \
	cs_edge_detect_v2 cs_copy_box cs_motion_report cs_config \
	cs_perm_generic cs_matrix cs_complgrngn cs_vector cs_random \
	cs_domultivec cs_sparser cs_compGrad_x cs_buffer cs_comp_step \
	cuda_test cs_test cs_ipcam cs_dct cs_decode_parser cs_quantize \
	cs_webcam cs_video_io cs_motion_detect_v3 cs_motion_detect_v4 \
	cs_mean_sd cs_recon cs_recon_thread)

EXE_C_SRCS:=readrl.c
EXE_CUDA_SRCS:= pheader.cu cs_test.cu cuda_test.cu 

CS_RECON_NO_THRD_SRCS:=cs_recon.cu
CS_RECON_NO_THRD_LIB:=$(TST_LIB_DIR)/librecon_nothrd$(LIB_EXT)
$(CS_RECON_NO_THRD_LIB): $(patsubst %.cu,$(TST_OBJ_DIR)/%$(OBJ_EXT), \
    $(CS_RECON_NO_THRD_SRCS))

CS_RECON_THRD_SRCS:=   cs_recon_thread.cu
CS_RECON_THRD_LIB:=$(TST_LIB_DIR)/librecon_thrd$(LIB_EXT)
$(CS_RECON_THRD_LIB): $(patsubst %.cu,$(TST_OBJ_DIR)/%$(OBJ_EXT), \
    $(CS_RECON_THRD_SRCS))

CS_MISC_LIB_FILES:= $(TST_OBJ_DIR)/cs_image$(OBJ_EXT) \
   $(patsubst %.cu,$(TST_OBJ_DIR)/%$(OBJ_EXT), \
  $(filter-out $(EXE_CUDA_SRCS) $(CS_RECON_NO_THRD_SRCS) $(CS_RECON_THRD_SRCS),\
  $(TST_CUDA_SRCS)))
CS_MISC_LIB:=$(TST_LIB_DIR)/libcs_misc$(LIB_EXT)
$(CS_MISC_LIB): $(CS_MISC_LIB_FILES)

TST_LIB_FILES:=$(CS_MISC_LIB) $(CS_RECON_NO_THRD_LIB) $(CS_RECON_THRD_LIB)

TST_C_EXE_FILES:=$(patsubst %.c,$(TST_EXE_DIR)/%$(EXE_EXT), $(EXE_C_SRCS))
TST_CUDA_EXE_FILES:=$(patsubst %.cu,$(TST_EXE_DIR)/%$(EXE_EXT), $(EXE_CUDA_SRCS))

CS_WHM_ENC_LIB:=$(TST_LIB_DIR)/libcs_whmenc$(LIB_EXT)

ifneq ($(wildcard /usr/lib/libopencv_core.so ),)
OPENCV_DIR:=/usr/lib
OPENCV_NAMES:= \
    libopencv_calib3d libopencv_core libopencv_features2d libopencv_flann \
    libopencv_highgui libopencv_imgproc libopencv_ml \
    libopencv_objdetect libopencv_photo libopencv_stitching \
    libopencv_superres libopencv_video libopencv_videoio libopencv_videostab
else
OPENCV_DIR:=/usr/local/lib
OPENCV_NAMES:= \
    libopencv_calib3d libopencv_contrib libopencv_core libopencv_features2d libopencv_flann \
    libopencv_gpu libopencv_highgui libopencv_imgproc libopencv_legacy libopencv_ml \
    libopencv_nonfree libopencv_objdetect libopencv_photo libopencv_stitching \
    libopencv_superres libopencv_ts libopencv_video libopencv_videostab
endif

OPENCV_LIBS:=$(patsubst %,$(OPENCV_DIR)/%.so, $(OPENCV_NAMES))

TST_CC_SYS_LIBS:=-lm
TST_CUDA_SYS_LIBS:= -lm -lgomp /usr/local/lib/libjson-c.so $(OPENCV_LIBS) $(TST_CUDA_SYS_LIBS)

$(TST_EXE_DIR)/readrl$(EXE_EXT): $(TST_OBJ_DIR)/readrl$(OBJ_EXT) $(CS_WHM_ENC_LIB) 
$(TST_EXE_DIR)/pheader$(EXE_EXT): $(TST_OBJ_DIR)/pheader$(OBJ_EXT)	\
  $(CS_WHM_ENC_LIB) $(CS_MISC_LIB)		\
  $(TST_LIB_DIR)/libcgenrnd$(LIB_EXT)
$(TST_EXE_DIR)/cs_test$(EXE_EXT): $(TST_OBJ_DIR)/cs_test$(OBJ_EXT) \
  $(CS_WHM_ENC_LIB) $(CS_MISC_LIB) $(TST_LIB_DIR)/libcgenrnd$(LIB_EXT)
$(TST_EXE_DIR)/cuda_test$(EXE_EXT): $(TST_OBJ_DIR)/cuda_test$(OBJ_EXT) \
  $(CS_WHM_ENC_LIB) $(CS_MISC_LIB) $(TST_LIB_DIR)/libcgenrnd$(LIB_EXT)

include $(ROOT_DIR)/util/rules.mak

#
#all : readrl pheader cs_perm_mlseq.o cs_interpolate.o cs_expand.o \
#	cs_header.o cs_helper.o cs_dbg.o cs_block.o cs_perm_selection.o \
#	cs_edge_detect.o cs_motion_detect.o cs_motion_detect_v2.o \
#	cs_edge_detect_v2.o cs_copy_box.o cs_motion_report.o cs_config.o \
#	cs_perm_generic.o cs_matrix.o cs_complgrngn.o cs_vector.o cs_random.o \
#	cs_domultivec.o cs_sparser.o cs_compGrad_x.o cs_buffer.o cs_comp_step.o \
#	cuda_test cs_ipcam.o cs_dct.o cs_decode_parser.o cs_quantize.o \
#	cs_webcam.o cs_video_io.o cs_motion_detect_v3.o cs_motion_detect_v4.o \
#	cs_mean_sd.o cs_image.o cs_recon.o cs_recon_thread.o
#
#NVCC_OPTS1:=-arch=compute_35 --compiler-bindir $(CUDA_CC) $(INCLUDE_OPT)
#ifneq ($(filter kanas%,$(HOSTNAME)),)
#NVCC_OPTS1+= -DOPENCV_3
#endif
#NVCC_OPTS:= $(NVCC_OPTS1) -dc
#
#RND_INCLUDE_OPT:=-I../../../RndStrmC -I../../../RndStrmC/cgen/lib
#
#readrl: readrl.c
#	cc readrl.c -lm -o readrl
#
#cs_config.o : cs_config.cu cs_config.h
#	nvcc $(NVCC_OPTS) cs_config.cu
#
#cs_comp_step.o : cs_comp_step.cu cs_comp_step.h cs_decode_misc.h
#	nvcc $(NVCC_OPTS) cs_comp_step.cu
#
#cs_decode_parser.o : cs_decode_parser.cu cs_decode_parser.h
#	nvcc $(NVCC_OPTS) cs_decode_parser.cu
#
#cs_buffer.o : cs_buffer.cu cs_buffer.h
#	nvcc $(NVCC_OPTS) cs_buffer.cu
#
#cs_random.o : cs_random.cu cs_random.h
#	nvcc $(NVCC_OPTS) cs_random.cu
#
#cs_domultivec.o : cs_domultivec.cu cs_domultivec.h
#	nvcc $(NVCC_OPTS) cs_domultivec.cu
#
#cs_compGrad_x.o : cs_compGrad_x.cu cs_compGrad_x.h cs_decode_misc.h
#	nvcc $(NVCC_OPTS) cs_compGrad_x.cu
#
#cs_sparser.o : cs_sparser.cu cs_sparser.h cs_decode_misc.h
#	nvcc $(NVCC_OPTS) cs_sparser.cu
#
#cs_mean_sd.o : cs_mean_sd.cu cs_mean_sd.h
#	nvcc $(NVCC_OPTS) cs_mean_sd.cu
#
#cs_recon_thread.o : cs_recon_thread.cu cs_recon.h
#	nvcc $(NVCC_OPTS) cs_recon_thread.cu
#
#cs_recon.o : cs_recon.cu cs_recon.h
#	nvcc $(NVCC_OPTS) cs_recon.cu
#
#cs_quantize.o : cs_quantize.cu cs_quantize.h
#	nvcc $(NVCC_OPTS) cs_quantize.cu
#
#cs_vector.o : cs_vector.cu cs_vector.h
#	nvcc $(NVCC_OPTS) cs_vector.cu
#
#cs_complgrngn.o : cs_complgrngn.cu cs_complgrngn.h cs_decode_misc.h
#	nvcc $(NVCC_OPTS) cs_complgrngn.cu
#
#cs_motion_report.o : cs_motion_report.cu cs_motion_report.h
#	nvcc $(NVCC_OPTS) cs_motion_report.cu
#
#cs_interpolate.o : cs_interpolate.cu cs_interpolate.h cs_dbg.h
#	nvcc $(NVCC_OPTS) cs_interpolate.cu
#
#cs_matrix.o : cs_matrix.cu cs_matrix.h cs_dbg.h
#	nvcc $(NVCC_OPTS) cs_matrix.cu
#
#cs_motion_detect_v2.o : cs_motion_detect_v2.cu cs_motion_detect_v2.h cs_dbg.h
#	nvcc $(NVCC_OPTS) cs_motion_detect_v2.cu
#
#cs_motion_detect_v3.o : cs_motion_detect_v3.cu cs_motion_detect_v3.h cs_dbg.h
#	nvcc $(NVCC_OPTS) cs_motion_detect_v3.cu
#
#cs_motion_detect_v4.o : cs_motion_detect_v4.cu cs_motion_detect_v4.h cs_dbg.h
#	nvcc $(NVCC_OPTS) cs_motion_detect_v4.cu
#
#cs_perm_generic.o : cs_perm_generic.cu cs_perm_generic.h cs_dbg.h
#	nvcc $(NVCC_OPTS) cs_perm_generic.cu
#
#cs_motion_detect.o : cs_motion_detect.cu cs_motion_detect.h cs_dbg.h
#	nvcc $(NVCC_OPTS) cs_motion_detect.cu
#
#cs_copy_box.o : cs_copy_box.cu cs_copy_box.h cs_dbg.h
#	nvcc $(NVCC_OPTS) cs_copy_box.cu
#
#cs_edge_detect_v2.o : cs_edge_detect_v2.cu cs_edge_detect_v2.h cs_dbg.h cs_copy_box.h
#	nvcc $(NVCC_OPTS) cs_edge_detect_v2.cu
#
#cs_edge_detect.o : cs_edge_detect.cu cs_edge_detect.h cs_dbg.h cs_copy_box.h
#	nvcc $(NVCC_OPTS) cs_edge_detect.cu
#
#cs_perm_mlseq.o : cs_perm_mlseq.cu cs_perm_mlseq.h cs_dbg.h
#	nvcc $(NVCC_OPTS) cs_perm_mlseq.cu
#
#cs_ipcam.o : cs_ipcam.cu cs_ipcam.h
#	nvcc $(NVCC_OPTS) cs_ipcam.cu
#
#cs_image.o : cs_image.cc  cs_image.h
#	g++ -I/usr/local/include/opencv -I/usr/local/include -c cs_image.cc
#
#cs_video_io.o : cs_video_io.cu cs_video_io.h
#	nvcc $(NVCC_OPTS) cs_video_io.cu
#
#cs_webcam.o : cs_webcam.cu cs_webcam.h
#	nvcc $(NVCC_OPTS) cs_webcam.cu
#
#cs_expand.o : cs_expand.cu cs_expand.h
#	nvcc $(NVCC_OPTS) cs_expand.cu
#
#cs_block.o : cs_block.cu cs_block.h
#	nvcc $(NVCC_OPTS) cs_block.cu
#
#cs_dbg.o : cs_dbg.cu cs_dbg.h
#	nvcc $(NVCC_OPTS) cs_dbg.cu
#
#cs_header.o : cs_header.cu cs_header.h
#	nvcc $(NVCC_OPTS) cs_header.cu
#
#cs_perm_selection.o : cs_perm_selection.cu cs_perm_selection.h
#	nvcc $(NVCC_OPTS) cs_perm_selection.cu
#
#cs_helper.o : cs_helper.cu cs_helper.h
#	nvcc $(NVCC_OPTS) cs_helper.cu
#
#cs_dct.o : cs_dct.cu cs_dct.h
#	nvcc $(NVCC_OPTS) cs_dct.cu
#
#pheader: pheader.cu cs_header.h ./cs_header.o
#	nvcc $(NVCC_OPTS1) pheader.cu ./cs_header.o -o pheader
#
#cs_test : cs_test.cu cs_interpolate.o cs_dbg.o cs_helper.o \
#	cs_header.o cs_copy_box.o cs_block.o cs_perm_mlseq.o cs_expand.o \
#	cs_perm_selection.o cs_motion_detect.o cs_edge_detect.o \
#	cs_edge_detect_v2.o cs_motion_detect_v2.o cs_ipcam.o cs_webcam.o
#	nvcc $(NVCC_OPTS1) cs_test.cu \
#		cs_dbg.o \
#		cs_helper.o \
#		cs_header.o \
#		cs_block.o \
#		cs_perm_mlseq.o \
#		cs_expand.o \
#		cs_interpolate.o \
#		cs_perm_selection.o \
#		cs_copy_box.o \
#		cs_motion_detect.o \
#		cs_motion_detect_v2.o \
#		cs_motion_detect_v3.o \
#		cs_edge_detect.o \
#		cs_edge_detect_v2.o \
#		cs_ipcam.o \
#		cs_webcam.o \
#		/usr/local/lib/libopencv_calib3d.so /usr/local/lib/libopencv_contrib.so /usr/local/lib/libopencv_core.so /usr/local/lib/libopencv_features2d.so /usr/local/lib/libopencv_flann.so /usr/local/lib/libopencv_gpu.so /usr/local/lib/libopencv_highgui.so /usr/local/lib/libopencv_imgproc.so /usr/local/lib/libopencv_legacy.so /usr/local/lib/libopencv_ml.so /usr/local/lib/libopencv_nonfree.so /usr/local/lib/libopencv_objdetect.so /usr/local/lib/libopencv_photo.so /usr/local/lib/libopencv_stitching.so /usr/local/lib/libopencv_superres.so /usr/local/lib/libopencv_ts.so /usr/local/lib/libopencv_video.so /usr/local/lib/libopencv_videostab.so \
#		-lm -lgomp -o cs_test
#
#cuda_test : cuda_test.cu cs_interpolate.o cs_dbg.o cs_helper.o \
#	cs_header.o cs_copy_box.o cs_block.o cs_perm_mlseq.o cs_expand.o \
#	cs_perm_selection.o cs_motion_detect.o cs_edge_detect.o \
#	cs_edge_detect_v2.o cs_motion_detect_v2.o
#	nvcc $(NVCC_OPTS1) cuda_test.cu \
#		cs_dbg.o \
#		cs_helper.o \
#		cs_header.o \
#		cs_block.o \
#		cs_perm_mlseq.o \
#		cs_expand.o \
#		cs_interpolate.o \
#		cs_perm_selection.o \
#		cs_copy_box.o \
#		cs_motion_detect.o \
#		cs_motion_detect_v2.o \
#		cs_edge_detect.o \
#		cs_edge_detect_v2.o \
#		-lm -lgomp -o cuda_test
#
#clean:
#	rm -f readrl pheader cs_perm_mlseq.o cs_interpolate.o cs_expand.o \
#	cs_header.o cs_helper.o cs_dbg.o cs_block.o cs_perm_selection.o \
#	cs_edge_detect.o cs_motion_detect.o cs_motion_detect_v2.o \
#	cs_edge_detect_v2.o cs_copy_box.o cs_motion_report.o cs_config.o \
#	cs_perm_generic.o cs_matrix.o cs_complgrngn.o cs_vector.o cs_random.o \
#	cs_domultivec.o cs_sparser.o cs_compGrad_x.o cs_buffer.o cs_comp_step.o \
#	cuda_test cs_ipcam.o cs_dct.o cs_test cs_webcam.o cs_video_io.o \
#	cs_motion_detect_v3.o cs_motion_detect_v4.o cs_decode_parser.o \
#	cs_mean_sd.o cs_recon.o cs_recon_thread.o

