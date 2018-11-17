all : readrl pheader cs_perm_mlseq.o cs_interpolate.o cs_expand.o \
	cs_header.o cs_helper.o cs_dbg.o cs_block.o cs_perm_selection.o \
	cs_edge_detect.o cs_motion_detect.o cs_motion_detect_v2.o \
	cs_edge_detect_v2.o cs_copy_box.o cs_motion_report.o cs_config.o \
	cs_perm_generic.o cs_matrix.o cs_complgrngn.o cs_vector.o cs_random.o \
	cs_domultivec.o cs_sparser.o cs_compGrad_x.o cs_buffer.o cs_comp_step.o \
	cuda_test cs_ipcam.o cs_dct.o cs_decode_parser.o cs_quantize.o \
	cs_webcam.o cs_video_io.o cs_motion_detect_v3.o cs_motion_detect_v4.o \
	cs_mean_sd.o cs_image.o cs_recon.o cs_recon_thread.o

NVCC_OPTS=-arch=sm_20 -dc
NVCC_OPTS1=-arch=sm_20

pheader: pheader.cu cs_header.h ./cs_header.o
	nvcc $(NVCC_OPTS1) --compiler-bindir $(CUDA_CC) pheader.cu ./cs_header.o -o pheader

readrl: readrl.c
	cc readrl.c -lm -o readrl

cs_config.o : cs_config.cu cs_config.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I/usr/local/include/json -c cs_config.cu

cs_comp_step.o : cs_comp_step.cu cs_comp_step.h cs_decode_misc.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I../../RndStrmC -I../../RndStrmC/cgen/lib \
		-I/usr/local/include/json -c cs_comp_step.cu

cs_decode_parser.o : cs_decode_parser.cu cs_decode_parser.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I../../RndStrmC -I../../RndStrmC/cgen/lib \
		-I/usr/local/include/json -c cs_decode_parser.cu

cs_buffer.o : cs_buffer.cu cs_buffer.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I/usr/local/include/json -c cs_buffer.cu

cs_random.o : cs_random.cu cs_random.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I../../RndStrmC -I../../RndStrmC/cgen/lib -c cs_random.cu

cs_domultivec.o : cs_domultivec.cu cs_domultivec.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I../cs_whm_encode -c cs_domultivec.cu

cs_compGrad_x.o : cs_compGrad_x.cu cs_compGrad_x.h cs_decode_misc.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I../cs_whm_encode -I../../RndStrmC \
		-I../../RndStrmC/cgen/lib -c cs_compGrad_x.cu

cs_sparser.o : cs_sparser.cu cs_sparser.h cs_decode_misc.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I../../RndStrmC \
		-I../../RndStrmC/cgen/lib -c cs_sparser.cu

cs_mean_sd.o : cs_mean_sd.cu cs_mean_sd.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_mean_sd.cu

cs_recon_thread.o : cs_recon_thread.cu cs_recon.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I../cs_whm_encode -c cs_recon_thread.cu

cs_recon.o : cs_recon.cu cs_recon.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I../cs_whm_encode -c cs_recon.cu

cs_quantize.o : cs_quantize.cu cs_quantize.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_quantize.cu

cs_vector.o : cs_vector.cu cs_vector.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_vector.cu

cs_complgrngn.o : cs_complgrngn.cu cs_complgrngn.h cs_decode_misc.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I../../RndStrmC -I../../RndStrmC/cgen/lib \
		-c cs_complgrngn.cu

cs_motion_report.o : cs_motion_report.cu cs_motion_report.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_motion_report.cu

cs_interpolate.o : cs_interpolate.cu cs_interpolate.h cs_dbg.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_interpolate.cu

cs_matrix.o : cs_matrix.cu cs_matrix.h cs_dbg.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_matrix.cu

cs_motion_detect_v2.o : cs_motion_detect_v2.cu cs_motion_detect_v2.h cs_dbg.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_motion_detect_v2.cu

cs_motion_detect_v3.o : cs_motion_detect_v3.cu cs_motion_detect_v3.h cs_dbg.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_motion_detect_v3.cu

cs_motion_detect_v4.o : cs_motion_detect_v4.cu cs_motion_detect_v4.h cs_dbg.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_motion_detect_v4.cu

cs_perm_generic.o : cs_perm_generic.cu cs_perm_generic.h cs_dbg.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_perm_generic.cu

cs_motion_detect.o : cs_motion_detect.cu cs_motion_detect.h cs_dbg.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_motion_detect.cu

cs_copy_box.o : cs_copy_box.cu cs_copy_box.h cs_dbg.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_copy_box.cu

cs_edge_detect_v2.o : cs_edge_detect_v2.cu cs_edge_detect_v2.h cs_dbg.h cs_copy_box.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_edge_detect_v2.cu

cs_edge_detect.o : cs_edge_detect.cu cs_edge_detect.h cs_dbg.h cs_copy_box.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_edge_detect.cu

cs_perm_mlseq.o : cs_perm_mlseq.cu cs_perm_mlseq.h cs_dbg.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_perm_mlseq.cu

cs_ipcam.o : cs_ipcam.cu cs_ipcam.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I/usr/local/include/opencv -I/usr/local/include -c cs_ipcam.cu

cs_image.o : cs_image.c cs_image.h
	g++ -I/usr/local/include/opencv -I/usr/local/include -c cs_image.c

cs_video_io.o : cs_video_io.cu cs_video_io.h
	nvcc $(NVCC_OPTS) -DOPENCV_3 --compiler-bindir $(CUDA_CC) -I/usr/local/include/opencv -I/usr/local/include -c cs_video_io.cu

cs_webcam.o : cs_webcam.cu cs_webcam.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -I/usr/local/include/opencv -I/usr/local/include -c cs_webcam.cu

cs_expand.o : cs_expand.cu cs_expand.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_expand.cu

cs_block.o : cs_block.cu cs_block.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_block.cu

cs_dbg.o : cs_dbg.cu cs_dbg.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_dbg.cu

cs_header.o : cs_header.cu cs_header.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_header.cu

cs_perm_selection.o : cs_perm_selection.cu cs_perm_selection.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_perm_selection.cu

cs_helper.o : cs_helper.cu cs_helper.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_helper.cu

cs_dct.o : cs_dct.cu cs_dct.h
	nvcc $(NVCC_OPTS) --compiler-bindir $(CUDA_CC) -c cs_dct.cu

cs_test : cs_test.cu cs_interpolate.o cs_dbg.o cs_helper.o \
	cs_header.o cs_copy_box.o cs_block.o cs_perm_mlseq.o cs_expand.o \
	cs_perm_selection.o cs_motion_detect.o cs_edge_detect.o \
	cs_edge_detect_v2.o cs_motion_detect_v2.o cs_ipcam.o cs_webcam.o
	nvcc $(NVCC_OPTS1) --compiler-bindir $(CUDA_CC) cs_test.cu \
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
		cs_motion_detect_v3.o \
		cs_edge_detect.o \
		cs_edge_detect_v2.o \
		cs_ipcam.o \
		cs_webcam.o \
		/usr/local/lib/libopencv_calib3d.so /usr/local/lib/libopencv_contrib.so /usr/local/lib/libopencv_core.so /usr/local/lib/libopencv_features2d.so /usr/local/lib/libopencv_flann.so /usr/local/lib/libopencv_gpu.so /usr/local/lib/libopencv_highgui.so /usr/local/lib/libopencv_imgproc.so /usr/local/lib/libopencv_legacy.so /usr/local/lib/libopencv_ml.so /usr/local/lib/libopencv_nonfree.so /usr/local/lib/libopencv_objdetect.so /usr/local/lib/libopencv_photo.so /usr/local/lib/libopencv_stitching.so /usr/local/lib/libopencv_superres.so /usr/local/lib/libopencv_ts.so /usr/local/lib/libopencv_video.so /usr/local/lib/libopencv_videostab.so \
		-lm -lgomp -o cs_test

cuda_test : cuda_test.cu cs_interpolate.o cs_dbg.o cs_helper.o \
	cs_header.o cs_copy_box.o cs_block.o cs_perm_mlseq.o cs_expand.o \
	cs_perm_selection.o cs_motion_detect.o cs_edge_detect.o \
	cs_edge_detect_v2.o cs_motion_detect_v2.o
	nvcc $(NVCC_OPTS1) --compiler-bindir $(CUDA_CC) cuda_test.cu \
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
	rm -f readrl pheader cs_perm_mlseq.o cs_interpolate.o cs_expand.o \
	cs_header.o cs_helper.o cs_dbg.o cs_block.o cs_perm_selection.o \
	cs_edge_detect.o cs_motion_detect.o cs_motion_detect_v2.o \
	cs_edge_detect_v2.o cs_copy_box.o cs_motion_report.o cs_config.o \
	cs_perm_generic.o cs_matrix.o cs_complgrngn.o cs_vector.o cs_random.o \
	cs_domultivec.o cs_sparser.o cs_compGrad_x.o cs_buffer.o cs_comp_step.o \
	cuda_test cs_ipcam.o cs_dct.o cs_test cs_webcam.o cs_video_io.o \
	cs_motion_detect_v3.o cs_motion_detect_v4.o cs_decode_parser.o \
	cs_mean_sd.o cs_recon.o cs_recon_thread.o

