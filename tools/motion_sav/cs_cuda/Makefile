INCLUDE_DIRS:= . cs_whm_encode cs_misc ../../RndStrmC ../../RndStrC/cgen/lib
INCLUDE_OPT:=$(patsubst %, -I%,$(INCLUDE_DIRS))

include $(ROOT_DIR)/util/defs.mak

RND_LIB:=$(TST_LIB_DIR)/libcgenrnd.a
all : yuvb420pcs

yuvb420pcs : yuvb420pcs.cu cs_misc/cs_interpolate.o cs_misc/cs_dbg.o cs_misc/cs_helper.o \
	cs_misc/cs_header.o cs_whm_encode/cs_whm_encode_b.o cs_misc/cs_block.o \
	cs_misc/cs_copy_box.o cs_misc/cs_edge_detect_v2.o cs_misc/cs_motion_detect_v2.o \
	cs_misc/cs_perm_mlseq.o cs_misc/cs_expand.o cs_misc/cs_perm_selection.o \
	cs_misc/cs_config.o cs_misc/cs_motion_report.o cs_misc/cs_dct.o cs_misc/cs_ipcam.o 

	nvcc --compiler-bindir $(CUDA_CC) $(INCLUDE_PATH) yuvb420pcs.cu \
		cs_misc/cs_dbg.o \
		cs_misc/cs_helper.o \
		cs_misc/cs_header.o \
		cs_misc/cs_block.o \
		cs_misc/cs_perm_mlseq.o \
		cs_misc/cs_expand.o \
		cs_misc/cs_interpolate.o \
		cs_misc/cs_perm_selection.o \
		cs_misc/cs_copy_box.o \
		cs_misc/cs_edge_detect_v2.o \
		cs_misc/cs_motion_detect_v2.o \
		cs_misc/cs_motion_report.o \
		cs_misc/cs_config.o \
		cs_misc/cs_ipcam.o \
		cs_misc/cs_dct.o \
		cs_whm_encode/cs_whm_encode_b.o \
		$(TST_LIB_DIR)/libcgenrnd.a \
		/usr/local/lib/libopencv_calib3d.so \
		/usr/local/lib/libopencv_contrib.so \
		/usr/local/lib/libopencv_core.so \
		/usr/local/lib/libopencv_features2d.so \
		/usr/local/lib/libopencv_flann.so \
		/usr/local/lib/libopencv_gpu.so \
		/usr/local/lib/libopencv_highgui.so \
		/usr/local/lib/libopencv_imgproc.so \
		/usr/local/lib/libopencv_legacy.so \
		/usr/local/lib/libopencv_ml.so \
		/usr/local/lib/libopencv_nonfree.so \
		/usr/local/lib/libopencv_objdetect.so \
		/usr/local/lib/libopencv_photo.so \
		/usr/local/lib/libopencv_stitching.so \
		/usr/local/lib/libopencv_superres.so \
		/usr/local/lib/libopencv_ts.so \
		/usr/local/lib/libopencv_video.so \
		/usr/local/lib/libopencv_videostab.so \
		-ljson-c -lm -lgomp -o yuvb420pcs

clean :
	rm -f yuvb420pcs
