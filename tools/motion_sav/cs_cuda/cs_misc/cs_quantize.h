#ifndef __CS_QUANTIZE_H__
#define __CS_QUANTIZE_H__

void h_do_unquan_adj_index ( int *d_in, int tbl_size, int noclip, int index_adj, int max_out ) ;
void h_do_unquan_msrmnts( float *d_inout, int tbl_size, float ampl, float intvl, float mean ) ;

int h_ck_bin ( int *d_in, int size, int max, int *zeroed, int skip ) ;

// quant

// dc is for no clips ... 
int
h_do_quant( float *dp, int *d_out, int blk_size, struct cube *h_cubep, struct cube *d_cubep,
	int nblk_in_x, int nblk_in_y,
	float *d_meanp, float *s_sdp,
	int *d_max_binp, int *d_num_binp, float *d_amplp, float *d_offset,
    float *d_dcp ) ;

#endif 
