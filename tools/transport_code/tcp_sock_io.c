#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>

#ifdef MATLAB_MEX_FILE
#include <mex.h>
#define printf mexPrintf
#endif

#include "tcp_sock_io.h"

#define IP_STR_SIZE (64)

static void set_sock_addr_in(struct sockaddr_in *addr,
			     int host_required,
			     const char *host,
			     unsigned long port,
			     const char **err_str
			     );
socket_t 
openTCPSocketServer(const char *lcl_host, 
		    unsigned long lcl_port,
		    size_t hostlen,
		    char *host,
		    unsigned long *port,
		    double timeout,
		    const char **err_str
		    )
{
  socket_t sockfd, sock;
  struct sockaddr_in addr;
  SOCKLEN_T addrlen = sizeof(addr);
  char ip_str[IP_STR_SIZE];
  const char *estr;
  char lhost[IP_STR_SIZE];
  pollfd_t fds[1];
  unsigned long upoll_wait;
  int upoll_result;
  
  if (timeout > (double)ULONG_MAX || timeout==UPOLL_INDEFINITE) {
    *err_str = "Timeout is too large or has a reserved value";
    return SOCKET_T_UNDEF;
  }
  upoll_wait = timeout? (unsigned long) floor(timeout*1E6 + 0.5): UPOLL_INDEFINITE; 

  if (lcl_host == NULL) {
    ip_to_str(INADDR_ANY, lhost);
    lcl_host = lhost;
  }

  if (err_str != NULL)
    *err_str = NULL;		/* Initially assume no error */
  else
    err_str = &estr;

  INIT_IP_STACK();

  /* Open a TCP socket (an Internet stream socket) */
  if ( (sockfd = socket(AF_INET, SOCK_STREAM, 0)) ==  SOCKET_T_UNDEF){
    get_sock_error(NULL, err_str);
    return sockfd;
  }

  set_sock_addr_in(&addr, 0, lcl_host, lcl_port, err_str);
  if (err_str !=NULL && *err_str != NULL) {
    CLOSESOCKET(sockfd);
    return SOCKET_T_UNDEF;
  }

  if (bind(sockfd, (struct sockaddr *) &addr, sizeof(addr)) == SOCKET_ERROR) {
    get_sock_error(NULL, err_str);
    CLOSESOCKET(sockfd);
    return SOCKET_T_UNDEF;
  }

  if (listen(sockfd, 1) == SOCKET_ERROR) {
    get_sock_error(NULL, err_str);
    CLOSESOCKET(sockfd);
    return SOCKET_T_UNDEF;
  }

  fds[0].fd = sockfd;
  fds[0].events = (POLLREAD | POLLFAIL);

  upoll_result = upoll(fds, 1, upoll_wait, NULL, err_str);
  if(upoll_result > 0 && (fds[0].revents & POLLFAIL)) {
    get_sock_error(NULL, err_str);
    upoll_result = -1;
  }
  else if(upoll_result == 0) {
    *err_str = "Timeout waiting for client to connect";
    upoll_result = -1;
  }
  if(upoll_result < 0) {
    CLOSESOCKET(sockfd);
    return SOCKET_T_UNDEF;
  }
    
  sock = accept(sockfd, (struct sockaddr *) &addr, &addrlen);

  if (sock == SOCKET_T_UNDEF) {
    get_sock_error(NULL, err_str);
    sock =  SOCKET_T_UNDEF;
  }
  else {

    if (host != NULL) {
      ip_to_str(ntohl(addr.sin_addr.s_addr), ip_str);
      if (hostlen <= strlen(ip_str)) {
	if (err_str != NULL) 
	  *err_str = "Not enough space for client IP address";
	CLOSESOCKET(sock);
      }
      else 
	strcpy(host, ip_str);
    }
    if (port != NULL)
      *port = (unsigned long) ntohs(addr.sin_port);
  }
  CLOSESOCKET(sockfd);

  return sock;
}

socket_t 
openTCPSocketClient(const char *host,
		    unsigned long port,
		    const char *lcl_host, 
		    unsigned long lcl_port,
		    const char **err_str
		    )
{
  socket_t sockfd;
  struct sockaddr_in addr;
  const char *estr;

  char lhost[IP_STR_SIZE];
  
  if (lcl_host == NULL) {
    ip_to_str(INADDR_ANY, lhost);
    lcl_host = lhost;
  }

  if (err_str != NULL)
    *err_str = NULL;		/* Initially assume no error */
  else
    err_str = &estr;

  INIT_IP_STACK();

  /* Open a TCP socket (an Internet stream socket) */
  if ( (sockfd = socket(AF_INET, SOCK_STREAM, 0)) ==  SOCKET_T_UNDEF){
    get_sock_error(NULL, err_str);
    return sockfd;
  }

  if (lcl_host == NULL)
    lcl_host = "localhost";

  set_sock_addr_in(&addr, 0, lcl_host, lcl_port, err_str);
  if (*err_str != NULL) {
    CLOSESOCKET(sockfd);
    return SOCKET_T_UNDEF;
  }

  if (bind(sockfd, (struct sockaddr *) &addr, sizeof(addr)) == SOCKET_ERROR) {
    get_sock_error(NULL, err_str);
    CLOSESOCKET(sockfd);
    return SOCKET_T_UNDEF;
  }

  set_sock_addr_in(&addr, 0, host, port, err_str);
  if (*err_str != NULL) {
    CLOSESOCKET(sockfd);
    return SOCKET_T_UNDEF;
  }
  if (connect(sockfd, (struct sockaddr *)&addr, sizeof(addr)) == SOCKET_ERROR) {
    get_sock_error(NULL, err_str);
    CLOSESOCKET(sockfd);
    sockfd = SOCKET_T_UNDEF;
  }
  return sockfd;
  
}

int sendTCPSocket(socket_t sock,
		  size_t datalen,
		  const void *data,
		  const char **err_str
		  )
{
  int ret_val = 0;
  const char *err_s = "";
  const char **perr = (err_str != NULL)? err_str: &err_s;

  if(send(sock, data, (int)datalen, 0) == SOCKET_ERROR)
    get_sock_error(&ret_val, perr);
  else
    *perr = NULL;

  return ret_val;
}

SSIZE_T recvTCPSocket(socket_t sock,
		      size_t datalen,
		      void *data,
		      double timeout,
		      const char **err_str
		      )
{
  SSIZE_T ret_val;
  pollfd_t fds[1];
  unsigned long upoll_wait;
  int upoll_result;
  const char *err_s = "";
  const char **perr = (err_str != NULL)? err_str: &err_s;

  if (timeout > (double)ULONG_MAX || timeout==UPOLL_INDEFINITE) {
    *perr = "Timeout is too large or has a reserved value";
    return SOCKET_ERROR;
  }
  upoll_wait = timeout? (unsigned long) floor(timeout*1E6 + 0.5): UPOLL_INDEFINITE; 

  fds[0].fd = sock;
  fds[0].events = (POLLREAD | POLLFAIL);
  upoll_result = upoll(fds, 1, upoll_wait, NULL, perr);
  if(upoll_result > 0) {
    if(fds[0].revents & POLLREAD) {
      /* Try to read */
      ret_val = recv(sock, data, (int)datalen, 0);
      if (ret_val == 0) {	/* Orderly shut down by peer */
	ret_val = SOCKET_ERROR;
	*perr = NULL;
      }
      else if(ret_val == SOCKET_ERROR) {
	get_sock_error(NULL, perr);
      }
      else {
	*perr = NULL;
      }
    } 
    else {			 /* Socket error */
      ret_val = SOCKET_ERROR;
      get_sock_error(NULL, perr);
    }
  }
  else if(upoll_result == 0) { /* time out */
    ret_val = 0;
    *perr = NULL;
  }
  else {			/* upoll_result < 0 */
    ret_val = SOCKET_ERROR;
    get_sock_error(NULL, perr);
  }

  return ret_val;
}

int closeTCPSocket(socket_t sock,
		   int timeout,
		   const char **err_str
		   )
{
  int ret_val = 0;
  const char *err_s = "";
  const char **perr = (err_str != NULL)? err_str: &err_s;
  struct linger so_linger;

  if(timeout < 0) {
    so_linger.l_onoff = 0;
    so_linger.l_linger = 0;
  }
  else {
    so_linger.l_onoff = 1;
    so_linger.l_linger = timeout;
  }

  if(setsockopt(sock, SOL_SOCKET, SO_LINGER, (void *)&so_linger, sizeof(so_linger))
     == SOCKET_ERROR) {
    get_sock_error(&ret_val, perr);
    return ret_val;
  }
  {
    socklen_t optlen = sizeof(so_linger);
    getsockopt(sock, SOL_SOCKET, SO_LINGER, (void *)&so_linger, &optlen);
  }

  if(CLOSESOCKET(sock) == SOCKET_ERROR)
    get_sock_error(&ret_val, perr);
  else
    *perr = NULL;

  return ret_val;
}

static void set_sock_addr_in(struct sockaddr_in *addr,
			     int host_required,
			     const char *host,
			     unsigned long port,
			     const char **err_str
			     )
{
  in_addr_t host_addr;
  in_port_t port_num;

  host_addr = name_to_ip(host, NULL);
  if (host_addr == INADDR_NONE) {
    *err_str = "Could not resolve address";
    return;
  }

  if (host_required && host_addr == INADDR_ANY) {
    *err_str = "Need a specific IP address";
    return;
  }

  port_num = (in_port_t) port;
  if(port != (unsigned long) port_num) {
    *err_str = "Illegal port number";
    return;
  }

  memset(addr, 0, sizeof(*addr));
  addr->sin_family = AF_INET;
  addr->sin_addr.s_addr = htonl(host_addr);
  addr->sin_port = htons(port_num);

  *err_str = NULL;
}

