#include "./sel_1m_idx.h"
#include "./sel_4m_idx.h"
#include "./sel_256k_idx.h"
#include "./tables.h"

int *selection_tbl[] = {
	sel_1m_idx,
	sel_4m_idx, 
	sel_256k_idx
} ;
