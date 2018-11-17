#ifndef __CS_DECODE_PARSER_H__
#define __CS_DECODE_PARSER_H__

struct vhc {
	int v ;
	int h ;
	int t ;
} ;

//1
#define CS_EncParams_code_len		2000
#define NUM_CS_EncParams_FLOAT	4
struct CS_EncParams {
	int n_frames ;
	int start_frame ;
	int random_seed ;	// use the one comes with SensingMatrixWH
	int process_color ;	// 1 : means color yuv
	int wnd_type ;
	int qntzr_outrange_action ;	// 1 : set to 0
	int lossless_coder ;	// 1 : no arithmatic coding
	int conv_mode ;
	int leng_conv_rng ;
	int random_rpt_spatial;	// 1 : repeat the seed from blk to blk in the same frames
	int random_rpt_temporal;	// 1 : repeat the seed from frames to frames
	struct vhc conv_rng ; // 1 1 1
	struct vhc blk_size ; // 72 88 8
	struct vhc blk_ovrlp ; // 0 0 0
	struct vhc zero_ext_b ; // 0 0 0
	struct vhc zero_ext_f ; // 0 0 0
	struct vhc wrap_ext ; // 0 0 0
	struct vhc blk_pre_diff ; // 0 0 0
	int case_no ;
	int n_cases ;

	// the following 3 items only for encoder ... decoder gets them from the per-block params
	float msrmnt_input_ratio ;	// compression ratio
	float qntzr_wdth_mltplr ;
	float qntzr_ampl_stddev ;	// sigma, std distribution

	float lossless_code_AC_gaus_thrsh ;	// for arithmatic coding only

	char msrmnt_mtrx_code[ CS_EncParams_code_len + 1] ;
} ;

// 2
struct RawVidInfo {
	int UV_present ;
	int precision ;	// 1 : 8 bits/pixel
	int n_frames ;
	int width ;	// 352
	int height ;	// 288
	int seg_start_frame ;
	int seg_n_frames ;
	int fps ;
	int uv_ratio [3] ; // only valid when UV_present is 1
		// 288/144:352/176:8/8 => 2:2:1	
} ;

// 3
struct VidRegion {
	int n_blk ;
	int blk_v ;
	int blk_h ;
	int blk_t ;
} ;

#define SensingMatrixWH_code_len	100

// 4
struct SensingMatrixWH {
	int n_rows ;	// # of measurements	72x88x8x1.5 * msrmnt_input_ratio = 7604
					// 1.5 is yuv 420 format
	int n_cols ;	// # of pixels ... 72x88x8 x3 = 152064
	int is_transposed ; 
	int seed ;
	char code[ SensingMatrixWH_code_len + 1] ;
	int sqr_order ;		// max_log2 of ( n_cols ), 262144
} ;

//5 
#define NUM_UniformQuantizer_FLOAT		3

struct UniformQuantizer {
	int save_clipped ;		// 0 : do not save ... len_s is 0
	float q_wdth_mltplr ;
	float q_wdth_unit ;
	float q_ampl_mltplr ;

	// derived

	float q_wdth ;
} ;

#define NUM_QuantMeasurementsBasic_FLOAT 2
//6
struct QuantMeasurementsBasic {
	int nbin ;
	int noclip ;
	int lenb ;
	int lens ;
	float mean_msr ;
	float stdv_msr ;

	int *h_msr_idxp ;	// should be allocated by the caller ... 

	// derived ...

	float *d_msr_p ;	// copy from h_msr_idxp to device
} ;


int get_UniformQuantizer( void *vp ) ;
int get_VidRegion( void *vp ) ;
int get_RawVidInfo( void *vp ) ;
int get_SensingMatrixWH( void *vp ) ;
int get_CS_EncParams( void *vp ) ;
int get_QuantMeasurementsBasic( void *vcsp ) ;

int cs_decode_parser_init( char *fname, int size ) ;
int get_next_element ( int type, void *d ) ;
int get_next_type ( int *type ) ;	// 1: good 0:err -1:eof
int p_element ( int type, char *s, void *d ) ; 
void cs_decode_parser_reinit ( int size ) ;

#endif 
