#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <math.h>
#include "cs_dbg.h"
#include "cs_cuda.h"
#include "cs_helper.h"

#include "cs_dbg.h"
#include "cs_helper.h"
#include "cs_whm_encode_b.h"
#include "cs_perm_generic.h"
#include "cs_domultivec.h"

#define CUDA_DBG
#define CUDA_DBG1

/*
h_do_multi_vec:

	the transformation is based on walsh-hadamard transform

	d_input : input data
	d_output : output data
	d_tmp : tmp buffer
	R/L_perm_tbl : permutation table
	keep_size : -1 ... keep all 

	NOTE: all d_xxx sizes have to be the same for all table,
	and has to be exactly the size of some power of 2

	tbl_size is the vhtc size, need not to be mod2

	return: data in d_output, d_input is not changed
*/

void
h_do_multi_vec ( float *d_input, float *d_output, float *d_tmp, int *d_R_perm_tbl,
	int *d_L_perm_tbl, int tbl_size, int keep )
{
	int size ;

	size = max_log2( tbl_size ) ;

#ifdef CUDA_DBG 
	printf("%s: din %p dout %p dtmp %p Rperm %p Lperm %p tblsize %d size %d\n",
		__func__, d_input, d_output, d_tmp, d_R_perm_tbl, d_L_perm_tbl, tbl_size, size ) ;
#endif 

	clear_device_mem( d_input + tbl_size, size - tbl_size ) ;
			
#ifdef CUDA_DBG 
	dbg_p_d_data_i("Right perm", d_R_perm_tbl, size ) ; 
	dbg_p_d_data_f("input", d_input, size ) ; 
#endif 

	h_do_permutation_generic_f1( d_input, d_tmp, d_R_perm_tbl, size ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f("input after R perm", d_tmp, size ) ; 
#endif 

	cs_whm_measurement_b( d_tmp, size, size ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f("input after whm", d_tmp, size ) ; 
	dbg_p_d_data_i("Left perm", d_R_perm_tbl, size ) ; 
#endif 

	h_do_permutation_generic_f2( d_tmp, d_output, d_L_perm_tbl, size ) ;

	if ( keep >= 0 )
		clear_device_mem( d_output + keep, size - keep ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f("input after L perm", d_tmp, size ) ; 
	dbg_p_d_data_f("output after L perm", d_output, size ) ; 
#endif 
}


/*
h_do_multi_trnsp_vec:

	the transformation is based on walsh-hadamard transform

	d_input : input data // 4752 ... filled 0 for the rest
	d_output : output data
	R/L_perm_tbl : permutation table

	size: has to be the same for all table, and has to be exactly the size of some power of 2

return: data in d_input
*/

void
h_do_multi_trnsp_vec ( float *d_input, float *d_output, int *d_R_perm_tbl, int *d_L_perm_tbl,
	int tbl_size )
{
	if ( tbl_size == max_log2( tbl_size ))
	{
		printf("%s: din %p dout %p Rperm %p Lperm %p tblsize %d\n", __func__,
			d_input, d_output, d_R_perm_tbl, d_L_perm_tbl, tbl_size ) ;
	}

#ifdef CUDA_DBG 
	printf("%s: din %p dout %p Rperm %p Lperm %p tblsize %d\n", __func__,
		d_input, d_output, d_R_perm_tbl, d_L_perm_tbl, tbl_size ) ;
#endif 

#ifdef CUDA_DBG 
	dbg_p_d_data_i("Right perm", d_R_perm_tbl, tbl_size ) ; 
	dbg_p_d_data_f("input", d_input, tbl_size ) ; 
#endif 

	h_do_permutation_generic_f1( d_input, d_output, d_L_perm_tbl, tbl_size ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f("input after R perm", d_output, tbl_size ) ; 
#endif 

	cs_iwhm_measurement_b( d_output, tbl_size, tbl_size ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f("input after R perm", d_output, tbl_size ) ; 

	dbg_p_d_data_i("Left perm", d_R_perm_tbl, tbl_size ) ; 
#endif 

	h_do_permutation_generic_f2( d_output, d_input, d_R_perm_tbl, tbl_size ) ;

#ifdef CUDA_DBG 
	dbg_p_d_data_f("input after L perm", d_input, tbl_size ) ; 
#endif 
}
