#include <stdio.h>
#include <math.h>

#define CUDA_MAX_THREADS_P_BLK 512

int
max_log2( int i ) 
{
	int k, j ;

	k = ( int )log2(( double )i ) ;
	j = (int)pow(2.0, k ) ;

	if ( j < i )
		j = (int)pow(2.0, k + 1 ) ;

	return ( j ) ;
}

int 
find_thread_blk ( int threads )
{
	int i, k, j ;

	k = CUDA_MAX_THREADS_P_BLK * 2 ;

	if ( threads < k )
	{
		j = max_log2 ( threads ) ;

		printf("j %d \n", j ) ;

		return ( j / 2 ) ;

#ifdef CUDA_OBS 

		i = max_log2( threads ) ;
		j = max_log2 ( threads / 2 ) ;

		if (( i - threads ) > ( threads - j ))
			return ( j / 2 ) ;
		else
			return ( i / 2 ) ;
#endif 
	}  
	return ( k / 2 ) ;
}

main()
{
	int i, j ;

	while ( 1 )
	{
		printf("==> ") ;
		scanf("%d", &i ) ;

		j = find_thread_blk ( i ) ;

		printf("i %d j %d \n", i, j ) ;
	}
}
