#include <stdio.h>

#define SIZE	16
int data[ SIZE * SIZE ] ;
int data1[ SIZE * SIZE ] ;

void
p_num_nm ( char *s, int *dp, int col, int row )
{    
	int i, j, *fp ;

	printf("%s : %s dp %p col %d row %d \n", __func__, s, dp, col, row ) ;

	fp = dp ;
	for ( i = 0 ; i < row ; i++ )
	{
		for ( j = 0 ; j < col ; j++ )
			printf("%d ", *fp++ ) ;
		printf("\n") ;
	}
}

reshape ( int *tp, int *fp, int size )
{
	int from, to, i, j ;

	for ( i = 0 ; i < size ; i++ )
	{	
		for ( j = 0 ; j < size ; j++ )
		{
			from = i * size + j ;
			to = j * size + i ;

			// printf("from %d to %d \n", from, to ) ;

			tp[ to ] = fp [ from ] ;
		}

	}
}

main()
{
	int i ;

	for ( i = 0 ; i < SIZE * SIZE ; i++ )
	{
		data[i] = i + 1 ;
		data1[i] = -1 ;
	}

	p_num_nm ( "before", data, SIZE, SIZE ) ;

	reshape ( data1, data, SIZE ) ;

	p_num_nm ( "after", data1, SIZE, SIZE ) ;

	return ( 0 ) ;
}

