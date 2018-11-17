/** Test tcp_sock_io.h */

#include <stdlib.h>
#include <stddef.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "tcp_sock_io.h"

#define MAX_HOST_NAME (256)
#define MAX_MSG (255)

static void help(const char *cmd);

static void
parse_addr_arg(const char * addr_str, /**< address specified as [<host>[:[<port>]]] */
	       int port_required, /**< if true port must be non-zero */
	       char host[MAX_HOST_NAME], /**< returns host name */
	       unsigned long *port /** returns port number (host order) */
	       );

int main(int argc, char *argv[]) {
  double timeout, recv_timeout;
  int linger = -1;
  unsigned int is_srvr;
  unsigned int is_sndr;
  char host[MAX_HOST_NAME], lcl_host[MAX_HOST_NAME];
  unsigned long port, lcl_port;
  char *eptr;
  socket_t sock;
  const char *err_str;
  char msg[MAX_MSG+1];
  unsigned char msg_len;
  int exit_status = EXIT_SUCCESS;

  if (argc == 1) { 
    help(argv[0]);
    exit(EXIT_SUCCESS);
  }
  
  timeout = strtod(argv[1], &eptr);
  if (errno || *eptr != '\0') {
      fprintf(stderr, "Illegal srvr_timeout argument: \"%s\"\n", argv[1]);
      exit(EXIT_FAILURE);
  }
  is_srvr = (timeout >= 0);

  if ((is_srvr && argc != 4) || (!is_srvr && (argc<4 || argc>5))) {
    fprintf(stderr, "Illegal number of arguments\n");
    exit(EXIT_FAILURE);
  }

  recv_timeout = strtod(&argv[2][1], &eptr);
  if (errno || *eptr != '\0') {
      fprintf(stderr, "Illegal value in recv_timeout argument: \"%s\"\n", &argv[2][1]);
      exit(EXIT_FAILURE);
  }

  switch(argv[2][0]) {
  case '>': is_sndr = 1;
    /* Set linger time */
    if(recv_timeout >= 0)
      linger = (int)ceil(recv_timeout); 
    break;
  case '<': is_sndr = 0;
    break;
  default:
    fprintf(stderr, "Unexpected prefix character in second argument\n");
    exit(EXIT_FAILURE);
  }

  if (is_srvr) {
    parse_addr_arg(argv[3], 1, lcl_host, &lcl_port);

    sock = openTCPSocketServer(lcl_host, lcl_port, sizeof(host), host, &port, timeout, &err_str);
    if (err_str != NULL) {
      fprintf(stderr, "openTCPSocketServer failed: %s\n", err_str);
      exit(EXIT_FAILURE);
    }
    
    fprintf(stderr, "\n*** Connected to %s:%lu\n\n", host,port);
  }
  else {
    parse_addr_arg(argv[3], 1, host, &port);
    if (argc == 5) {
      parse_addr_arg(argv[4], 0, lcl_host, &lcl_port);
    }
    else {
      ip_to_str(INADDR_ANY, lcl_host);
      lcl_port = 0;
    }

    sock = openTCPSocketClient(host, port, lcl_host, lcl_port, &err_str);
    if (err_str != NULL) {
      fprintf(stderr, "openTCPSocketClient failed: %s\n", err_str);
      exit(EXIT_FAILURE);
    }
  }

  if (is_sndr) {
    fprintf(stdout, ">> ");

    while(fgets(msg, sizeof(msg), stdin)) {
      msg_len = (unsigned char)strlen(msg);
      if (msg_len < 2)
	break;
      if(msg[msg_len-1] != '\n') {
	fprintf(stderr, "\n **** message too long ***\n");
	while(getchar() != '\n'); /* Flush rest of long line */
      }
      else if(sendTCPSocket(sock, sizeof(msg_len), &msg_len, &err_str) != 0 ||
	      sendTCPSocket(sock, msg_len, msg, &err_str) != 0) {
	fprintf(stderr, "Failed sending message: %s\n", err_str);
	exit_status = EXIT_FAILURE;
	break;
      }
      fprintf(stdout, ">> ");
    }
    if(exit_status == EXIT_SUCCESS) {
      msg_len = 0;
      if(sendTCPSocket(sock, sizeof(msg_len), &msg_len, &err_str) != 0) {
	fprintf(stderr, "Failed sending zero-length message: %s\n", err_str);
	exit_status = EXIT_FAILURE;
      }
    }
  }
  else {			/* receiver case */
    SSIZE_T rtvl;

    while ((rtvl=recvTCPSocket(sock, sizeof(msg_len), &msg_len, recv_timeout, &err_str)) > 0) {
      unsigned char mlen;

      if(msg_len == 0)
	break;

      mlen = msg_len;

      while (msg_len > 0) {
	SSIZE_T rcvd_len;

	rcvd_len = recvTCPSocket(sock, msg_len, msg, recv_timeout, &err_str);
	if(rcvd_len <= 0) {
	  exit_status = EXIT_FAILURE;

	  if(rcvd_len == 0)
	    fprintf(stderr, "\n *** Timed out after receiving %d bytes of %d bytes in message\n",
		    mlen - msg_len, mlen);
	  else if (err_str == NULL)
	    fprintf(stderr,
		    "\n *** Remote side closed gracefully after sending %d bytes of %d bytes in message\n",
		    mlen - msg_len, mlen);
	  else
	    fprintf(stderr,
		    "\n *** Error after receiving %d bytes of %d bytes in message: %s\n",
		    mlen - msg_len, mlen, err_str);
	  
	  goto terminate;
	}
	msg[rcvd_len] = '\0';
	fprintf(stdout, "%s\n", msg);
	fflush(stdout);

	msg_len -= rcvd_len;
      }
    }

  terminate:
    if (rtvl == 0)
      fprintf(stderr, "\n *** Timed out waiting for a message\n");
    else if (rtvl < 0) {
      if (err_str == NULL)
	fprintf(stderr, "\n *** Remote side closed gracefully\n");
      else {
	fprintf(stderr, "\n *** Failed receiving message length: %s\n", err_str);
	exit_status = EXIT_FAILURE;
      }
    }
  }

  closeTCPSocket(sock, linger, &err_str);
  if (exit_status==EXIT_SUCCESS && err_str != NULL) {
    fprintf(stderr, "closeTCPSocket failed: %s\n", err_str);
    exit(EXIT_FAILURE);
  }

  exit(exit_status);
      
  
}

static void help(const char *cmd) {
  printf("%s <is_srvr> <is_sndr> [<addr>]:<port> [[<lcl_addr>][:[<lcl_port>]]]\n"
	 "Args:\n"
	 "  srvr_timeout - (double) if non-negative open as a server (waiting for connection).\n"
         "                 The value indicates the duration after which the server times out\n"
	 "                 (0 means wait indefinitely). If srvr_timeout is negative open as\n"
         "                 a client (initiating connection)\n"
	 "  recv_timeout - One of the characters \'<\' or '> followed by a value (double).\n"
	 "                 You may need to use single quotes or backslash to prevent interpretation\n"
         "                 of these characters by the shell.\n"
         "                 If the prefix character is \'<\' this program recives data, in which case\n"
         "                 the value indicates the duration after which the receiver times out\n"
	 "                 (0 means wait indefinitely). If the prefix character is '>' this \n"
         "                 program sends data and the value indicates the time to linger after\n"
         "                 closing (-1=use default). non-integer values are rounded up \n"
	 "  addr - If server, optional local address to bind to. If client, server's address\n"
	 "  port - If server, local port to bind to, if client, server's port\n"
	 "  lcl_addr - If server, should not be specified. If client, optional"
	 " interface to bind to\n"
	 "  lcl_port - If server, should not be specified. If client, optional"
	 " port to bind to\n",
	 cmd
	 );
}

/** Parse an address argument  */
static void
parse_addr_arg(const char * addr_str,
	       int port_required,
	       char host[MAX_HOST_NAME],
	       unsigned long *port)
{
  char *eptr, *col_ptr;

  strncpy(host, addr_str, MAX_HOST_NAME);
  col_ptr = strrchr(addr_str, ':');
  if (col_ptr == NULL) {
    *port = 0;
  } else {
    ptrdiff_t col_pos = col_ptr - addr_str;

    if (col_pos < MAX_HOST_NAME)
      memset(host+col_pos, '\0', MAX_HOST_NAME - col_pos);

    if (addr_str[col_pos+1] == '\0')
      *port = 0;
    else {
      *port = strtoul(&addr_str[col_pos+1], &eptr, 0);
      if (errno || *eptr != '\0') {
	fprintf(stderr, "Illegal port in \"%s\"\n", addr_str);
	exit(EXIT_FAILURE);
      }
    }
  }

  if (host[MAX_HOST_NAME-1] != '\0') {
    fprintf(stderr, "Host name too long in \"%s\"\n", addr_str);
    exit(EXIT_FAILURE);
  }

  if (port_required && *port == 0) {
    fprintf(stderr, "No non-zer port in \"%s\"\n", addr_str);
    exit(EXIT_FAILURE);
  }

  if (host[0] == '\0')
    strcpy(host, "INADDR_ANY");
}
