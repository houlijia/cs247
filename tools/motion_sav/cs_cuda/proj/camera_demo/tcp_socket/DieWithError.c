#include <stdio.h>  /* for perror() */
#include <stdlib.h> /* for exit() */

#include "tcp_socket.h"

void DieWithError(const char *errorMessage)
{
	perror(errorMessage);
		exit(1);
}
