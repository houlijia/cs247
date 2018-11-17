#include <stdio.h>
#include "do_resize.h"

unsigned char *do_bicubic ( unsigned char *, int, int, int, int ) ;

unsigned char input[] = {
14,     7,     5,    12,    15,     6,    10,     4,     5,     7,    13,    11,     8,    10,    15,     2,
15,    15,     1,    13,    16,    14,     8,    15,    13,     2,     7,    12,     7,    10,    16,     5,
3,    13,     2,     5,     9,    10,     1,     3,     7,     4,     4,    11,     8,     4,     8,    13,
15,    16,    14,    11,     3,     9,     6,    14,    15,     2,     7,     8,     5,     5,     2,     1,
11,    11,    12,    11,     3,    15,     3,     9,     3,     3,     2,     9,     9,     8,     5,    15,
2,     1,     6,     3,     5,     5,    13,    16,     5,     4,     3,     5,     9,     4,     7,    12,
5,    14,    16,     2,    14,    13,     5,     2,     3,     7,    16,    12,    14,    14,    10,     8,
9,    15,     1,     8,     5,    13,     9,     8,     3,     1,    16,     4,    13,     4,     5,    10,
16,    11,     8,    16,    14,     7,     3,     2,    14,    15,    10,    11,    11,     4,    10,     4,
16,    13,     7,     6,     4,    10,    10,    16,    10,    16,     1,     3,     7,     3,    12,     8,
3,    12,    13,    10,    15,     2,     5,     1,     9,     8,     4,     6,    13,     4,     4,    16,
16,     7,    13,     4,     6,     1,    11,    13,     3,     8,     6,    11,     9,     7,     2,     9,
16,    11,     3,    13,     4,     9,    12,    14,    14,     6,    14,    13,     6,     5,     5,     9,
8,     3,     8,     5,     5,    13,    12,    14,    10,    15,     1,     2,    16,    15,     6,     4,
13,    12,     8,     9,    10,    15,     8,     2,     6,     6,     1,    15,    15,     7,     7,     8,
3,     1,    11,    12,     8,     3,     2,     7,     9,     2,     3,    13,     9,     3,     9,    10
} ;

void
p_num_nm_f ( const char *s, float *dp, int col, int row )
{    
	int i, j ;
	float *fp ;

	printf("%s : %s dp %p col %d row %d \n", __func__, s, (void*)dp, col, row ) ;

	fp = dp ;
	for ( i = 0 ; i < row ; i++ )
	{
		for ( j = 0 ; j < col ; j++ )
			printf("%f ", *fp++ ) ;
		printf("\n") ;
	}
}

void
p_num_nm_uc ( const char *s, unsigned char *dp, int col, int row )
{    
	int i, j ;
	unsigned char *fp ;

	printf("%s : %s dp %p col %d row %d \n", __func__, s, dp, col, row ) ;

	fp = dp ;
	for ( i = 0 ; i < row ; i++ )
	{
		for ( j = 0 ; j < col ; j++ )
			printf("%d ", *fp++ ) ;
		printf("\n") ;
	}
}

void
p_num_nm ( const char *s, int *dp, int col, int row )
{    
	int i, j, *fp ;

	printf("%s : %s dp %p col %d row %d \n", __func__, s, (void*)dp, col, row ) ;

	fp = dp ;
	for ( i = 0 ; i < row ; i++ )
	{
		for ( j = 0 ; j < col ; j++ )
			printf("%d ", *fp++ ) ;
		printf("\n") ;
	}
}

#ifdef CUDA_OBS 

// 16x16

a =

14     7     5    12    15     6    10     4     5     7    13    11     8    10    15     2
15    15     1    13    16    14     8    15    13     2     7    12     7    10    16     5
3    13     2     5     9    10     1     3     7     4     4    11     8     4     8    13
15    16    14    11     3     9     6    14    15     2     7     8     5     5     2     1
11    11    12    11     3    15     3     9     3     3     2     9     9     8     5    15
2     1     6     3     5     5    13    16     5     4     3     5     9     4     7    12
5    14    16     2    14    13     5     2     3     7    16    12    14    14    10     8
9    15     1     8     5    13     9     8     3     1    16     4    13     4     5    10
16    11     8    16    14     7     3     2    14    15    10    11    11     4    10     4
16    13     7     6     4    10    10    16    10    16     1     3     7     3    12     8
3    12    13    10    15     2     5     1     9     8     4     6    13     4     4    16
16     7    13     4     6     1    11    13     3     8     6    11     9     7     2     9
16    11     3    13     4     9    12    14    14     6    14    13     6     5     5     9
8     3     8     5     5    13    12    14    10    15     1     2    16    15     6     4
13    12     8     9    10    15     8     2     6     6     1    15    15     7     7     8
3     1    11    12     8     3     2     7     9     2     3    13     9     3     9    10

should see //

p_num_nm : BOX dp 0x1506010 col 4 row 4
161 143 128 119
127 138 96 147
171 120 135 123
136 138 130 134
p_num_nm : BOX dp 0x1506050 col 4 row 4
3 33 12 27
-13 0 -28 -9
15 12 25 -5
-2 28 20 -2
p_num_nm : BOX dp 0x1506090 col 4 row 4
35 21 -18 -5
9 8 -38 3
17 -2 31 -7
-2 -4 6 18
p_num_nm : BOX dp 0x15060d0 col 4 row 4
27 -9 8 -23
29 -8 -14 -9
11 -6 21 -11
4 -20 18 -22
p_num_nm : BOX dp 0x1506110 col 4 row 4
5 7 -14 -1
-23 -34 30 -21
21 10 29 -11
20 -38 24 18
p_num_nm : BOX dp 0x1506150 col 4 row 4
15 15 36 -21
-25 -14 -8 33
3 22 -1 31
22 -10 34 20


#endif 

int main()
{
	// float *tp ;
	unsigned char *tp ;

	setbuf( stdout, NULL ) ;
	setbuf( stderr, NULL ) ;

	p_num_nm_uc("data", input, 16, 16 ) ;

	// tp = do_resize( input, 16, 16, 10, 10 ) ;
	tp = do_bicubic( input, 16, 16, 10, 10 ) ;

	p_num_nm_uc("OUT", tp, 10, 10 ) ;
}

