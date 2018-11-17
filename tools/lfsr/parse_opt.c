/** \file parse_opt.c

(C) Copyright Alcatel-Lucent 2006
*/

#ifndef PARSE_OPT_C_GRD
#define	PARSE_OPT_C_GRD 

#include <string.h>

#include "parse_opt.h"

int
nextParseOpt(int argc,
			 char *argv[],
			 struct ParseOpt *po
	     ) /* returns option letter if successful or -1 otherwise */
{
	char *p;
	const char *q;				/* points to option letter */
	char * next_opt = NULL;
	int ret_val = -1;

	po->arg = NULL;				/* By default */
	if(po->next_opt != NULL)
		p = po->next_opt;
	else if(po->ind >= argc)
		return -1;
	else {
		p = argv[po->ind];	
		if(p[0] != '-' || p[1] == '\0')
			return -1;
		p++;
	}

	/* Search for arg and test validity */
	q = strchr(po->opt, *p);
	if(q == NULL);				/* Not a legal option - do nothing */
	else if(q[1] == ':') {		/* look for arg */
		if(p[1] != '\0') {
			ret_val = *p++;
			po->arg = p;
		}
		else if(po->ind + 1 < argc) {
			ret_val = *p;
			po->arg = argv[++(po->ind)];
		}
	}
	else {						/* option letter without an argument */
		ret_val = *p++;
		if(*p != '\0')		/* OK option - no extraneous characters  */
			next_opt = p;
	}

	if(ret_val != -1 && po->next_opt == NULL)
		po->ind += 1;

	po->next_opt = next_opt;

	return ret_val;
}

#endif	/* PARSE_OPT_C_GRD */

