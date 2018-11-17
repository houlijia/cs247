all : readrl pheader cs_perm_mlseq.o cs_interpolate.o cs_expand.o \
	cs_header.o cs_helper.o cs_dbg.o cs_block.o cs_perm_selection.o \
	cs_edge_detect.o cs_motion_detect.o cs_motion_detect_v2.o \
	cs_edge_detect_v2.o cs_copy_box.o cs_motion_report.o cs_config.o \
	cuda_test cs_ipcam.o cs_test

pheader: pheader.cu cs_header.h ./cs_header.o
	nvcc --compiler-bindir $(CUDA_CC) pheader.cu ./cs_header.o -o pheader

readrl: readrl.c
	cc readrl.c -lm -o readrl

cs_config.o : cs_config.cu cs_config.h
	nvcc --compiler-bindir $(CUDA_CC) -I/usr/local/include/json -c cs_config.cu

cs_motion_report.o : cs_motion_report.cu cs_motion_report.h
	nvcc --compiler-bindir $(CUDA_CC) -c cs_motion_report.cu

cs_interpolate.o : cs_interpolate.cu cs_interpolate.h cs_dbg.h
	nvcc --compiler-bindir $(CUDA_CC) -c cs_interpolate.cu

cs_motion_detect_v2.o : cs_motion_detect_v2.cu cs_motion_detect_v2.h cs_dbg.h
	nvcc --compiler-bindir $(CUDA_CC) -c cs_motion_detect_v2.cu

cs_motion_detect.o : cs_motion_detect.cu cs_motion_detect.h cs_dbg.h
	nvcc --compiler-bindir $(CUDA_CC) -c cs_motion_detect.cu

cs_copy_box.o : cs_copy_box.cu cs_copy_box.h cs_dbg.h
	nvcc --compiler-bindir $(CUDA_CC) -c cs_copy_box.cu

cs_edge_detect_v2.o : cs_edge_detect_v2.cu cs_edge_detect_v2.h cs_dbg.h cs_copy_box.h
	nvcc --compiler-bindir $(CUDA_CC) -c cs_edge_detect_v2.cu

cs_edge_detect.o : cs_edge_detect.cu cs_edge_detect.h cs_dbg.h cs_copy_box.h
	nvcc --compiler-bindir $(CUDA_CC) -c cs_edge_detect.cu

cs_perm_mlseq.o : cs_perm_mlseq.cu cs_perm_mlseq.h cs_dbg.h
	nvcc --compiler-bindir $(CUDA_CC) -c cs_perm_mlseq.cu

cs_ipcam.o : cs_ipcam.cu cs_ipcam.h
	nvcc --compiler-bindir $(CUDA_CC) -I/usr/local/include/opencv -I/usr/local/include -c cs_ipcam.cu

cs_expand.o : cs_expand.cu cs_expand.h
	nvcc --compiler-bindir $(CUDA_CC) -c cs_expand.cu

cs_block.o : cs_block.cu cs_block.h
	nvcc --compiler-bindir $(CUDA_CC) -c cs_block.cu

cs_dbg.o : cs_dbg.cu cs_dbg.h
	nvcc --compiler-bindir $(CUDA_CC) -c cs_dbg.cu

cs_header.o : cs_header.cu cs_header.h
	nvcc --compiler-bindir $(CUDA_CC) -c cs_header.cu

cs_perm_selection.o : cs_perm_selection.cu cs_perm_selection.h
	nvcc --compiler-bindir $(CUDA_CC) -c cs_perm_selection.cu

cs_helper.o : cs_helper.cu cs_helper.h
	nvcc --compiler-bindir $(CUDA_CC) -c cs_helper.cu

cs_test : cs_test.cu cs_interpolate.o cs_dbg.o cs_helper.o \
	cs_header.o cs_copy_box.o cs_block.o cs_perm_mlseq.o cs_expand.o \
	cs_perm_selection.o cs_motion_detect.o cs_edge_detect.o \
	cs_edge_detect_v2.o cs_motion_detect_v2.o cs_ipcam.o
	nvcc --compiler-bindir $(CUDA_CC) cs_test.cu \
		cs_dbg.o \
		cs_helper.o \
		cs_header.o \
		cs_block.o \
		cs_perm_mlseq.o \
		cs_expand.o \
		cs_interpolate.o \
		cs_perm_selection.o \
		cs_copy_box.o \
		cs_motion_detect.o \
		cs_motion_detect_v2.o \
		cs_edge_detect.o \
		cs_edge_detect_v2.o \
		cs_ipcam.o \
		/usr/local/lib/libopencv_calib3d.so /usr/local/lib/libopencv_contrib.so /usr/local/lib/libopencv_core.so /usr/local/lib/libopencv_features2d.so /usr/local/lib/libopencv_flann.so /usr/local/lib/libopencv_gpu.so /usr/local/lib/libopencv_highgui.so /usr/local/lib/libopencv_imgproc.so /usr/local/lib/libopencv_legacy.so /usr/local/lib/libopencv_ml.so /usr/local/lib/libopencv_nonfree.so /usr/local/lib/libopencv_objdetect.so /usr/local/lib/libopencv_photo.so /usr/local/lib/libopencv_stitching.so /usr/local/lib/libopencv_superres.so /usr/local/lib/libopencv_ts.so /usr/local/lib/libopencv_video.so /usr/local/lib/libopencv_videostab.so \
		-lm -lgomp -o cs_test

cuda_test : cuda_test.cu cs_interpolate.o cs_dbg.o cs_helper.o \
	cs_header.o cs_copy_box.o cs_block.o cs_perm_mlseq.o cs_expand.o \
	cs_perm_selection.o cs_motion_detect.o cs_edge_detect.o \
	cs_edge_detect_v2.o cs_motion_detect_v2.o
	nvcc --compiler-bindir $(CUDA_CC) cuda_test.cu \
		cs_dbg.o \
		cs_helper.o \
		cs_header.o \
		cs_block.o \
		cs_perm_mlseq.o \
		cs_expand.o \
		cs_interpolate.o \
		cs_perm_selection.o \
		cs_copy_box.o \
		cs_motion_detect.o \
		cs_motion_detect_v2.o \
		cs_edge_detect.o \
		cs_edge_detect_v2.o \
		-lm -lgomp -o cuda_test

clean:
	rm -f readrl pheader cs_block.o cs_edge_detect.o cs_helper.o cs_dbg.o cs_header.o\
		cs_motion_detect.o cs_copy_box.o cuda_test cs_motion_detect_v2.o \
		cs_test cs_edge_detect_v2.o cs_motion_report.o cs_ipcam.o
