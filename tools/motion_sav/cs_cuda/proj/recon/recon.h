#ifndef __RECON_H__
#define __RECON_H__

// cf epsilon@init_solver_eps.m

struct eps {
	float lgrng_chg ;
	float lgrng_chg_final ;
	float lgrng_chg_rate ;

	float A_maxerr ;
	float A_sqrerr ;

	float D_maxerr ;
	float D_sqrerr ;	// dx - w 
} ;


struct CS_DecParams {
	// int WND_HANNING: 1
	// int WND_TRIANGLE: 2
	// int wnd_types: [1 2]
	// int wnd_type: 1
	// int expand_cnstrnt_level: 0
	// int max_out_iters: 128
	// int max_int_iters: 4
	// int max_iters: 512
	// step_ratio_thrsh: 0.7500
	// eps_dLd: 1.0000e-04

	float eps_lgrng_chg_init ;
	float eps_lgrng_chg_rate ;
	float eps_lgrng_chg ;
	int eps_A_maxerr ;
	int eps_A_sqrerr ;
	float eps_D_maxerr ;
	float eps_D_sqrerr ;

	// int disp: 0
	// int disp_db: []
	// int init: 0
	// int solve_exact_fctr: 0
	// q_trans_Aerr: 0.0100
	// int q_trans_msrs_err: 1
	// ref: []
	// int use_old: 0
	// int Q_msrmnts: 0
	// sparsifier: [1x1 struct]

	int beta_rate ;
	float beta_rate_thresh ;
	float beta_A0 ;
	float beta_D0 ;
	float beta_A ;	// Inf
	float beta_D ;	// Inf

	// lsrch_c: 1.0000e-05
	// lsrch_alpha_rate: 0.6000
	// lsrch_wgt: 0.9900
	// lsrch_wgt_rate: 0.9000
	// int cmpr_frm_by_frm: 1
	// int cmpr_blk_by_blk: 1
	// int cmp_solve_stats: 0
	// int parallel: 0
	// int case_no: 0
	// int n_cases: 0
} ;

struct rc_SensingMatrixWH {
	// use_matlab_WHtransform: 0
	// log2order: 16
	// wh_mode: 1
	// wh_mode_names: {'hadamard'  'dyadic'  'sequency'}
	// unit_permut_L: 0
	// unit_permut_R: 0
	int sqr_order ;
	// IPL: [65536x1 double]
	// PR: [65536x1 double]
	// check_mult_eps: []
	// n_rows: 1901
	// n_cols: 38016
	// zeroed_rows: [4x1 double]
	// is_transposed: 0
	// code: 0
	// seed: 1000
	// rnd_type: 0
	// rnd_strm: [1x1 RandStream]
	// default_seed: 0
	// default_rnd_type: 'mlfg6331_64'
} ;

#endif 
