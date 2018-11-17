/** \file util/ip_name.c

(C) Copyright Alcatel-Lucent 2008
*/

/** ip_name.c Function for converting host name to IP  */
#include <string.h>
#include <ctype.h>

#if !defined(_MSC_VER) && !defined(__MINGW64__)
#include <netinet/in.h>
#include <netdb.h>
#endif

#include "tcp_sock_io.h"


/** Read an IP address represented by \c str. 
   \c str may contain the actual IP address, either as 0{x|X}\<hex no\> or in
   a dotted decimal notation (x.y.u.z).  Alternatively, \c str may contain the host
   name. In either case, if \c next is not NULL and \c str contains a colon (:),
   the last colon in the string and anything following it are ignored.

   \param str is host name string.
   \param next if not NULL, returns a pointer to the end of the decoded
   address string. Thus, if decoding failed it returns a pointer to str; else
   if str contains a column it returns a pointer to the final column; else it
   returns a pointer to the terminating null character.
   \return the IP address in host order or INADDR_NONE if the host was not found.
*/
in_addr_t
name_to_ip(const char *str,
		  char **next
		  )
{
	unsigned long ip=INADDR_NONE;
	char *p=(char *)str;
	char *n;
	struct hostent *hp;
	char name_buf[256];

#ifdef _WIN32
	if(!strcmp(str,"INADDR_ANY")) /* In Mex under windows gethostbyname fails on this */
	   str = "0.0.0.0";
#endif 
	ip = str_to_ip(str, &n);
	if(n == str) {
	  const char *q, *q0;

	  q0 = p+strlen(p)-1;
	  for(q = q0 ; q != p && isdigit((int)*q); q--);
	  if(q != p && next != NULL && *q == ':') {
	    if((unsigned)(q-p) < sizeof(name_buf)) {
	      memcpy(name_buf, str, q-p);
	      name_buf[q-p] = '\0';
	      p = name_buf;
	    } else {			/* Name too long */
	      if(next != NULL)
		*next = (char *)str;
	      return INADDR_NONE;
	    }
	  } else
	    q = q0+1;
	   
	  hp = gethostbyname(p);
	  if(hp  == NULL)
	    n = (char *)str;
	  else {
	    ip = ntohl(*(unsigned long *)hp->h_addr);
	    n = (char *)q;
	  }
	}


	if(next != NULL) 
	  *next = n;

	return ip;
}
