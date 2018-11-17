#include <stdio.h>
#include "../i_recon.h"
#include "../ibuf.h"
#include "../serial_wht3.h"
#include <stdlib.h>
#include <string.h>

#define ROW	5
#define COL	4

#define CUDA_DBG 

#define MEA_CNT	15

#define SIZE	( ROW * COL )

float ibuf[ SIZE ] = {
93.3993,   65.5478,    4.6171,   95.0222,
67.8735,   17.1187,    9.7132,    3.4446,
75.7740,   70.6046,   82.3458,   43.8744,
74.3132,    3.1833,   69.4829,   38.1558,
39.2227,   27.6923,   31.7099,   76.5517	 
} ;

#ifdef CUDA_MATLAB

phi_tbl

1 10 11 12 13 14 15 16
2 3 4 5 6 7 8 9
17 18 19 20 21 22 23 24
25 26 27 28 29 30 31 32
33 34 35 36 37 38 39 40
41 42 43 44 45 46 47 48
49 50 51 56 57 58 59 60
52 53 54 55 61 62 63 64

GAP

49 5 66 9 53 17 6 31
17 8 52 14 42 11 74 71
98 53 98 18 66 38 27 67
72 10 65 40 63 20 43 54
51 82 81 84 30 49 55 70
48 82 46 81 44 34 95 67
6 73 44 7 2 96 42 18
69 15 83 40 99 93 99 13

with 64 measurements should see 

48.300941       0.578340        -0.323842       -0.185253       1.643504        1.153964        0.090203        -0.884933         
-0.587181       -4.711256       -1.232493       0.395592        2.855857        2.681562        0.636727        0.610660          
-1.321056       -2.677945       -2.923937       3.461256        2.656100        2.150076        1.786619        1.122942        
-0.751664       -1.185684       -0.350937       0.511056        -1.746106       -0.767101       0.312161        0.398269         
-5.396724       5.305006        1.270186        8.867506        1.453125        9.086256        0.920976        -0.875518         
2.052973        0.105310        1.323563        -0.890625       -2.748591       -2.055087       1.476331        -0.163864         
-6.427973       -3.234375       2.648756        -2.406994       -2.804697       -2.121135       -2.106287       1.611408          
-4.221035       -4.062246       -3.298907       1.146723        -0.455384       -1.169968       0.486409        3.988441          

see pix

// MEA_CNT 64
p_num_nm_uc : PIXPIX dp 0x1a2bca0 col 3 row 36  
0 0 0 
0 0 0 
11 11 11 
80 80 80 
75 75 75 
18 18 18 
0 0 0 
0 0 0 
97 97 97 
75 75 75 
60 60 60 
50 50 50 
0 0 0 
0 0 0 
14 14 14 
0 0 0 
0 0 0 
9 9 9 
149 149 149 
36 36 36 
249 249 249 
41 41 41 
255 255 255 
26 26 26 
3 3 3 
37 37 37 
0 0 0 
0 0 0 
0 0 0 
41 41 41 
0 0 0 
74 74 74 
0 0 0 
0 0 0 
0 0 0 
0 0 0 

// MEA_CNT 15
p_num_nm_uc : PIXPIX dp 0x33a8ca0 col 3 row 36  
0 0 0 
14 14 14 
188 188 188 
99 99 99 
157 157 157 
145 145 145 
0 0 0 
0 0 0 
200 200 200 
104 104 104 
160 160 160 
131 131 131 
0 0 0 
0 0 0 
181 181 181 
111 111 111 
182 182 182 
196 196 196 
0 0 0 
0 0 0 
208 208 208 
118 118 118 
190 190 190 
182 182 182 
0 0 0 
0 0 0 
193 193 193 
134 134 134 
222 222 222 
253 253 253 
0 0 0 
0 0 0 
237 237 237 
160 160 160 
255 255 255 
253 253 253 



#endif 

// 1 relative
int phi_tbl[] = {
1, 10, 11, 12, 13, 14, 15, 16,
2, 3, 4, 5, 6, 7, 8, 9,
17, 18, 19, 20, 21, 22, 23, 24,
25, 26, 27, 28, 29, 30, 31, 32,
33, 34, 35, 36, 37, 38, 39, 40,
41, 42, 43, 44, 45, 46, 47, 48,
49, 50, 51, 56, 57, 58, 59, 60,
52, 53, 54, 55, 61, 62, 63, 64
} ;

int GAP_measurements[] = {
49,     5,    66,     9,    53,    17,     6,    31,
17,     8,    52,    14,    42,    11,    74,    71,
98,    53,    98,    18,    66,    38,    27,    67,
72,    10,    65,    40,    63,    20,    43,    54,
51,    82,    81,    84,    30,    49,    55,    70,
48,    82,    46,    81,    44,    34,    95,    67,
6,    73,    44,     7,     2,    96,    42,    18,
69,    15,    83,    40,    99,    93,    99,    13 } ;

float obuf[ SIZE ] ;

void GAP_test() ;
void TV_denoising_test() ;
void reconst_test() ;

main( int ac, char *av[] )
{
	int cmd ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	if ( ac < 2 )
	{
		printf("Usage : %s [1]=denoising [2]=GAP [3]=reconstruct \n", av[0]) ;
		exit( 1 ) ;
	}

	cmd = atoi ( av[1] ) ;

	if (( cmd <= 0 ) || ( cmd > 3 ))
	{
		printf("Usage2 : %s [1]=denoising [2]=GAP [3]=reconstruct cmd %d\n", av[0], cmd ) ;
		exit( 2 ) ;
	}

	switch( cmd ) {
	case 1 :
		TV_denoising_test() ;
		break ;
	case 2 :
		GAP_test() ;
		break ;
	case 3 :
		reconst_test() ;
		break ;
	default :
		printf("Usage1 : %s [1]=denoising [2]=GAP [3]=reconstruct\n", av[0]) ;
		exit( 2 ) ;
	}

	exit( 0 ) ;
}

void
reconst_test()
{
	float *fp ;
	int i, *GAP_measurements_1 ;
	unsigned char *cp ;

#ifdef CUDA_DBG 
	i_recon_set_dbg ( phi_tbl ) ;
#endif 

	// buf_init ( 64 * sizeof ( float ), 8 ) ;

	struct recon_param para = {
		8,
		6,
		6,
		10,
		1.23,
		2.45,
		1, 
		1,
		1,
		MEA_CNT,
   		0	} ;

	p_num_nm ("phi_tbl", phi_tbl, 8, 8 ) ;

	GAP_measurements_1 = ( int * )malloc ( sizeof ( int ) * 8 * 8 * 3 ) ; 

	for ( i = 0 ; i < 3 ; i++ )
	{
		memcpy ( GAP_measurements_1 + i * MEA_CNT, GAP_measurements, sizeof( int ) * MEA_CNT ) ; 	
#ifdef CUDA_OBS 
		*(GAP_measurements_1 + i * 8 * 8) = *(GAP_measurements_1 + i * 8 * 8) + i ;
#endif 
	}

	p_num_nm ("DATA", GAP_measurements_1, MEA_CNT, 3 ) ;

	buf_p("START") ;

	cp = reconstruct ( GAP_measurements_1, &para, &i ) ;

	printf("reconstruct return %p size %i \n", cp, i ) ;

	if ( cp != NULL )
	{
		p_num_nm_uc("PIXPIX", cp, 3, para.c * para.r ) ;
	}

	buf_p("ALL DONE") ;

}

void
GAP_test()
{
	float *fp ;

#ifdef CUDA_DBG 
	i_recon_set_dbg ( phi_tbl ) ;
#endif 

	buf_init ( 64 * sizeof ( float ), 8 ) ;

	struct recon_param para = {
		8,
		6,
		6,
		10,
		1.23,
		2.45,
		1, 
		1,
		1,
		MEA_CNT,
   		0	} ;

	p_num_nm ("phi_tbl", phi_tbl, 8, 8 ) ;

	buf_p("START") ;

	fp = TV_GAP_rgb_use( GAP_measurements, &para ) ;

	if ( fp )
		p_num_nm_f ("TV_GAP_rgb_use return", fp, 8, 8 ) ;

	buf_p("after TV_GAP_rgb_use") ;

}

void
TV_denoising_test()
{
	buf_init( SIZE * sizeof( float ), 8 ) ;

	buf_p("START") ;

	p_num_nm_f ( "IN", ibuf, COL, ROW ) ; 

	// float lambda = 1.23 ;
	buf_p("START 1") ;
	TV_denoising( ibuf, obuf, 1.23, COL, ROW, 10 ) ;
	// TV_denoising( ibuf, obuf, lambda, COL, ROW, 10 ) ;

	p_num_nm_f ( "OUT", obuf, COL, ROW ) ; 

	buf_p("DONE") ;
}

#ifdef CUDA_OBS 
TV_denoising ... should see

92.784302       65.240295       5.539599        94.407204
68.181000       17.733700       9.713200        4.367100
74.851501       70.604599       81.115799       43.566902
74.005707       4.413300        68.867905       39.078300
39.222698       27.999802       32.017403       75.936699


// for test 2 with 15 measurements ... should see

10.404228       -0.001847       0.154191        0.750217        0.391333        0.734307        -2.782652       -3.332425         
2.987890        -0.378798       0.056183        0.766285        0.406000        0.642074        0.592373        0.657495          
4.279233        -0.495267       -0.086278       0.815669        0.422906        0.653663        0.535721        0.450934          
2.987890        -0.705023       -0.104819       0.740218        0.451575        0.744168        0.799268        0.817755    
4.279233        -0.724561       -0.263611       0.849780        0.483481        0.776037        0.744452        0.666155          
2.987890        -0.951599       -0.243076       0.786950        0.547506        0.907242        1.032431        1.045186          
4.279233        -1.044084       -0.386254       0.967872        0.651953        1.040977        1.032940        0.872403          
-2.615863       -2.950418       0.039549        0.913968        0.770509        1.248887        3.047375        3.852983          


#endif 

