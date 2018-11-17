/* Copyright (C) 2012 Alcatel-Lucent */

/** \file read_prim_poly.c */

#include <stdio.h>
#include <errno.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>

#include "prim_poly.h"

int readPrimPolyList(const char *fname,
		     PrimPolyList *prpl_lst,
		     char *err_msg,
		     size_t err_msg_size
		     )
{
    int err = 0;
    FILE *inp;
    size_t prpl_len = prpl_lst->cnt;

    inp = fopen(fname, "rt");
    if(inp == NULL) {
        err = errno;
	errno = 0;
	snprintf(err_msg, err_msg_size,
		 "failed opening %s (%s)",fname,

		 strerror(err));
	return err;
    }

    for(;;) {
        unsigned long degree, pol, bpol = 0;
        char bits[128];
	int read_cnt;
	int len;
	int k;

	read_cnt = fscanf(inp, "  %lu %s %lx", &degree, bits, &pol);
	if (read_cnt == EOF) 
	    break;
	else if (read_cnt < 3) {
	    err = ferror(inp);
	    snprintf(err_msg, err_msg_size,
		     "Error reading file (%s)", strerror(err));
	    break;
	}
    
	len = strlen(bits);
	if (len != (int)(degree + 1) || bits[0] != '1') {
	    err = EOF-1;
	    snprintf(err_msg, err_msg_size,
		     "Degree (%ld) does not match string (%s)", degree, bits);
	    break;
	}

	for (k=1; k<len; k++) {
	    bpol = bpol << 1;
	    if (bits[k] == '0')
		continue;
	    else if (bits[k] == '1')
		bpol = bpol + 1;
	    else {
		err = EOF-2;
		snprintf(err_msg, err_msg_size,
			 "Degree (%ld) unexpected characters in string (%s)", 
			 degree, bits);
		goto finish;
	    }
	}
    
	if (bpol != pol) {
	    err = EOF-2;
	    snprintf(err_msg, err_msg_size,
		     "Degree (%ld) polynomial (0x%lX) does "
		     "not match string (%s=0x%lX)", 
		     degree, pol, bits, bpol);
	    break;
	}

	if(prpl_len <= prpl_lst->cnt) {
	    const size_t list_increment = 16;
	    PrimPoly *prv_prpl = prpl_lst->prpl;
	    size_t new_cnt = prpl_lst->cnt + list_increment;
      
	    prpl_lst->prpl = (PrimPoly *)realloc(prpl_lst->prpl,
						 new_cnt*sizeof(prpl_lst->prpl[0]));
	    if(prpl_lst->prpl == NULL){
		prpl_lst->prpl=prv_prpl;
		err = EOF-3;
		snprintf(err_msg, err_msg_size, "Malloc failed");
		break;
	    }
      
	    prpl_len = new_cnt;
	}
	{
	    PrimPoly primpl = {(u_32)pol, degree, NULL};

	    prpl_lst->prpl[prpl_lst->cnt++] = primpl;
	}
    }
      
 finish:
      
    return err;
}

