/* $Id: ip_util.c 16 2008-04-29 18:21:43Z rma $ */

/** \file util/ip_util.c IP utility functions */


#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>

#ifndef _WIN32
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/utsname.h>
#include <netdb.h>
#else
#include "WinSock2.h"
#include "ws2tcpip.h"
#endif

#include <fcntl.h>

#if defined(__linux) || defined(__linux__) || defined(linux)
#include <net/if.h>
#include <sys/ioctl.h>
#include <arpa/inet.h>
#endif

#include "tcp_sock_io.h"
#include "timeval.h"

#ifdef __ICC
#pragma warning (disable:981) /* "operands are evaluated in unspecified order" */
#endif

static in_addr_t my_ip_addr = INADDR_NONE;                      /* 0 indicates undefined */


#ifdef _WIN32
static WSADATA wsaData;
static int winsock_initialized=0;
static socket_t upoll_dummy_sock = SOCKET_T_UNDEF;

void init_winsock(void)
{
  WORD wVersionRequested = MAKEWORD(2,2);
  int err;
  const char *errstr;

  if(winsock_initialized)
    return;

  winsock_initialized = 1;

  err = WSAStartup(wVersionRequested, &wsaData);
  if(err) {
    get_sock_error(NULL, &errstr);
    fprintf(stderr, "IP Stack initialization failed. Err=%d (%s)\n",
            err, errstr);

    exit(EXIT_FAILURE);
  }

  upoll_dummy_sock = socket(PF_INET, SOCK_DGRAM, 0);
  if (upoll_dummy_sock == SOCKET_T_UNDEF) {
    get_sock_error(&err, &errstr);
    fprintf(stderr, "Failed to create upoll_dummy_sock. Err=%d (%s)\n",
            err, errstr);
    assert(0);                          /* Force failure */
  }

}
#endif

void
ip_to_str(in_addr_t ip,         /* host order */
          char *str
          )
{
  sprintf(str, IP_FMT, IP_PRNT(ip));
}

in_addr_t
str_to_ip(const char *str,
          char **next
          )                                             /* IP returned in host order */
{
  in_addr_t ip=0;
  int i;
  char *p=(char *)str;
  char *n;
  unsigned long val;

  if(str[0]=='0' && (str[1]=='x' || str[1]=='X')) {
    ip = strtoul(str+2,&n,16);
  } else {
    for(i=0;i<4;i++) {
      val = strtoul(p,&n,10);
      if (((i != 3) && (*n != '.')) ||
          (val > 255)
          ) {
        ip = INADDR_NONE;
        n = (char *)str;
        break;
      }
      ip |= val << (3-i)*CHAR_BIT;
      if(i<3)
        n++;
      p = n;
    }
  }

  if(next != NULL)
    *next = n;

  return ip;
}

void
setIPAddr(struct sockaddr *sadr,
          in_addr_t ip,         /* Host order */
          in_port_t port
          )
{
#if 1  /* Using this version because gcc 4.4 complains about breaking anti-aliasing rules... */
  struct sockaddr_in sadr_in;

  memset(&sadr_in,0,sizeof(sadr_in));
  sadr_in.sin_family = AF_INET;
  sadr_in.sin_port   = htons(port);
  sadr_in.sin_addr.s_addr = htonl(ip);
  memcpy(sadr, &sadr_in, sizeof(sadr_in));
#else
  struct sockaddr_in *sadr_in = (struct sockaddr_in *)(void *)sadr;

  memset(sadr,0,sizeof(*sadr));
  sadr->sa_family = AF_INET;
  sadr_in->sin_port = htons(port);
  sadr_in->sin_addr.s_addr = htonl(ip);
#endif
}

in_addr_t
getIPAddr(const struct sockaddr *sadr,
          in_port_t *port       /* If not NULL returns port number */
          )                                             /* returns IP address */
{
  const struct sockaddr_in *sadr_in = (const struct sockaddr_in *)sadr;

  if(port != NULL)
    *port = ntohs(sadr_in->sin_port);
  return ntohl(sadr_in->sin_addr.s_addr);
}

void
get_sock_error(int *err,         /**< If not NULL, returns error code */
               const char **err_str    /**< It not NULL returns error string,
					  or NULL if successful */
               )
{
  int err_val;

#ifdef _WIN32
  err_val = WSAGetLastError();
#else
  err_val = errno;
  errno = 0;
#endif

  if(err != NULL)
    *err = err_val;
  if(err_str != NULL) {
    if(!err_val)
      *err_str = NULL;
    else {
#ifdef __unix
      *err_str = strerror(err_val);
#elif defined(_WIN32)
      switch(err_val) {
	/* general errors */
      case WSANOTINITIALISED: *err_str = "WSANOTINITIALISED: Winsock not initialized";
	break;
      case WSAENETDOWN: *err_str = "WSAENETDOWN: network subsystem failed";
	break;
      case WSAEINPROGRESS: *err_str =
	  "WSAEINPROGRESS: blocking Winsock 1.1 call in progress "
	  "or still processing callback function";
	break;
      case WSAENOBUFS: *err_str = "WSAENOBUFS:- no buffer space available, "
	  "or in reading illegal buffers or fromlen too small";
	break;
      case WSAEFAULT: *err_str = "WSAEFAULT: command parameter not in user's address space "
	  "or bad WSADATA pointer";
	break;
      case WSAENOTSOCK: *err_str = "WSAENOTSOCK: not a socket";
	break;
      case WSAEINVAL: *err_str = "WSAEINVAL: socket already bound to an address, or invalid command, "
	  "or (for reading) socket not bound or len<=0";
	break;
      case WSAEWOULDBLOCK: *err_str = "WSAEWOULDBLOCK: would block but non blocking socket";
	break;
      case WSAEINTR: *err_str = "WSAEINTR: Blocking call cancelled through WSACancelBlockingCall";
	break;
      case WSAENETRESET: *err_str = "WSAENETRESET: connection time out when SO_KEEPALIVE was set";
	break;
      case WSAEAFNOSUPPORT: *err_str = "WSAEAFNOSUPPORT: address family not supported; or OOB not supportd";
	break;

	/* WSAStartup() errors */
      case WSASYSNOTREADY: *err_str =  "SASYSNOTREADY: Network subsystem not ready";
	break;
      case WSAVERNOTSUPPORTED: *err_str = "WSAVERNOTSUPPORTED: Could not find a supported version";
	break;
      case WSAEPROCLIM: *err_str = "WSAEPROCLIM: Exceeded limit on number of tasks supported";
	break;

	/* socket() errors */
      case WSAEMFILE: *err_str = "WSAEMFILE: no more socket descriptors";
	break;
      case WSAEPROTONOSUPPORT: *err_str = "WSAEPROTONOSUPPORT: protocol not supported";
	break;
      case WSAEPROTOTYPE: *err_str = "WSAEPROTOTYPE: wrong protocol for type of socket";
	break;
      case WSAESOCKTNOSUPPORT: *err_str = "WSAESOCKTNOSUPPORT: socket type not supported";
	break;

	/* bind() errors */
      case WSAEACCES: *err_str =
	  "WSAEACCES: connect to broadcast addr failed - "
	  "SO_BROADCAST not enabled";
	break;
      case WSAEADDRINUSE: *err_str = "WSAEADDRINUSE: address in use";
	break;
      case WSAEADDRNOTAVAIL: *err_str = "WSAEADDRNOTAVAIL: address not valid on this host";
	break;

	/* ioctlsocket() errors - no specific errors */

	/* WSAIoctl() errors */
      case WSA_IO_PENDING: *err_str = "WSA_IO_PENDING: overlapped operation initialized";
	break;
      case WSAEOPNOTSUPP: *err_str = "WSAEOPNOTSUPP: operation not supported";

	/* getsockopt() or setsockopt errors */
      case WSAENOPROTOOPT: *err_str = "WSAENOPROTOOPT: option unknown or not supported";
	break;

	/* recv or recvfrom errors */
      case WSAEISCONN: *err_str = "WSAEISCONN: function not allowed on connected socket";
	break;
      case WSAESHUTDOWN: *err_str = "WSAESHUTDOWN: socket has been shut down";
	break;
      case WSAEMSGSIZE: *err_str = "WSAEMSGSIZE: message too large - truncated";
	break;
      case WSAETIMEDOUT: *err_str = "WSAETIMEDOUT: network failure or remote side went down";
	break;
      case WSAECONNREFUSED: *err_str = "WSAECONNREFUSED: attempt to connect forcefully rejected";
	break;
      case WSAECONNRESET: *err_str = "WSAECONNRESET: hard close/abort by remote side, or "
	  "(UDP) previous send resulted in ICMP port unreachable";
	break;

	/* Unknown errors */
      default: *err_str = "Unknown Winsock error";
	break;
      }
#endif
    }
  }
  errno = 0;
}

/** Report a socket error as warning and return the error number.
    \param file - name of source file in which the error occurred
    \param line - line number in source file in which the error occurred
    \param fmt - Format of error message. Should have at most one field
    (string) for the error description.  If NULL a standard
    error description is given.
    \param report_fun - a function like warn or fail
*/
int
report_sock_err(const char *file, int line, const char *fmt,
                void (*report_fun)(const char *, int, int, const char *, ...)
                )
{
  int err;
  const char *err_str=NULL;

  get_sock_error(&err, &err_str);

  if(fmt==NULL)
    fmt = "I/O error (%s)";

  report_fun(file, line, err, fmt, err_str);

  return err;

}

#ifndef _WIN32
/** get the blocking status \return 1 if blocking, 0 if non-blocking,
    -1=error (errno should contain the error code).
*/
int
get_sock_blocking(socket_t sock
                  )     
{
  int val;

  val = fcntl(sock, F_GETFL);
  if(val == -1)
    return val;
  else
    return !(val & O_NONBLOCK);
}
#endif

/** make socket either blocking or non-blocking.
    \return 0 for success or -1 for failure (errno should contain the error code).
*/
int
set_sock_blocking(socket_t sock,
                  int block
                  )                             
{
#ifdef _WIN32
  unsigned long arg = !block;
  int err;

  err = ioctlsocket(sock, FIONBIO, &arg);
  if(err) {
    errno = err;
    err = -1;
  }

  return err;
#else
  long val;

  val = fcntl(sock, F_GETFL);
  if(val == -1) return val;

  if(block)
    val = fcntl(sock, F_SETFL, (val & ~O_NONBLOCK));
  else
    val = fcntl(sock, F_SETFL, (val | O_NONBLOCK));

  return (val==-1)? val: 0;
#endif
}

#if (defined(__CYGWIN__) || defined(_WIN32)) && !defined(MSG_DONTWAIT)
SSIZE_T recv_nowait(int s, void *buf, size_t len, int flags)
{
  SSIZE_T ret_val;
  pollfd_t pollfd[1];

  pollfd[0].fd = s;
  pollfd[0].events = POLLREAD;

  ret_val = upoll(pollfd, 1, 0, NULL, NULL);
  if(ret_val > 0)
    ret_val = recv(s, buf, len, flags);
  else
    ret_val = -1;

  return ret_val;
}
#endif

#if (defined(__CYGWIN__)  || defined(_WIN32)) && !defined(MSG_DONTWAIT)
SSIZE_T recvfrom_nowait(int s, void *buf, size_t len, int flags,
                        struct sockaddr *from, socklen_t *fromlen)
{
  SSIZE_T ret_val;
  pollfd_t pollfd[1];

  pollfd[0].fd = s;
  pollfd[0].events = POLLREAD;

  ret_val = upoll(pollfd, 1, 0, NULL, NULL);
  if(ret_val > 0)
    ret_val = recvfrom(s, buf, len, flags, from, fromlen);
  else
    ret_val = -1;

  return ret_val;
}
#endif

#ifndef _WIN32                                  /* Does not have recvmsg() */
#if (defined(__CYGWIN__) || defined(_WIN32)) && !defined(MSG_DONTWAIT)
SSIZE_T recvmsg_nowait(int s, struct msghdr *msg, int flags)
{
  SSIZE_T ret_val;
  pollfd_t pollfd[1];

  pollfd[0].fd = s;
  pollfd[0].events = POLLREAD;

  ret_val = upoll(pollfd, 1, 0, NULL, NULL);
  if(ret_val > 0)
    ret_val = recvmsg(s, msg, flags);
  else
    ret_val = -1;

  return ret_val;
}
#endif
#endif

unsigned
get_sock_mcast_ttl(socket_t sock,
                   int *err,
                   const char **err_str
                   ) /** \return returns the multicast time to live of the socket.
                         In case of error returns 0. The error code is in errno
                     */
{
  SOCKLEN_T len = 1;
#if defined(_WIN32)
  char val;
  const int level = IPPROTO_IP;
#else
  unsigned char val;
  const int level = SOL_IP;
#endif

  if(getsockopt(sock, level, IP_MULTICAST_TTL, &val, &len)) {
    val = 0;                            /* For error */
    get_sock_error(err,err_str);
  }
  return val;
}

int
set_sock_mcast_ttl(socket_t sock,
                   unsigned ttl,        /* Time to live for multicast datagrams sent from socket */
                   const char **err_str /* If != NULL returns error string,
                                           or NULL if successful */
                   )                    /* Returns 0 for success or an error code. */
{
  int err;
  SOCKLEN_T len = 1;
#if defined(_WIN32)
  char val;
  const int level = IPPROTO_IP;
#else
  unsigned char val;
  const int level = SOL_IP;
#endif
        
  if(ttl >= 256) {
    errno = ERANGE;
    return 0;
  }

  val = (unsigned char) ttl;
  if(setsockopt(sock, level, IP_MULTICAST_TTL, &val, len)) {
    get_sock_error(&err, err_str);
    return err;
  }
  return 0;
}

/* Force a self IP address.*/
void set_my_ip_addr(in_addr_t ip)
{
  my_ip_addr = htonl(ip);
}

#if defined(__linux) || defined(__linux__) || defined(linux)

/** Get IP address of an interface
    \return the IP address (host order) corresponding to an
    interface. If no IP is available INADDR_NONE is returned */
in_addr_t get_ifc_ip(const char *ifc)
{
  socket_t s;
  int rc;
  struct ifreq ifr;
  struct sockaddr_in *sin;
  in_addr_t ip;

  s = socket(PF_INET, SOCK_STREAM, 0);
  sin = (struct sockaddr_in *) &ifr.ifr_addr;
  strncpy(ifr.ifr_name, ifc, sizeof(ifr.ifr_name));

  rc = ioctl(s, SIOCGIFADDR, &ifr);
  CLOSESOCKET(s);

  if(rc < 0) return INADDR_NONE;

  memcpy(&ip, &sin->sin_addr, sizeof(ip));
  return ntohl(ip);
}

#endif /* defined(__linux) || defined(__linux__) || defined(linux) */

#if defined(__linux) || defined(__linux__) || defined(linux)
in_addr_t get_ifcn_ip(unsigned num){
  char ifc[16];

  sprintf(ifc,"eth%u", num);

  return get_ifc_ip(ifc);
}

#endif /* defined(__linux) || defined(__linux__) || defined(linux) */

in_addr_t get_frst_self_ip(get_self_ip_t *self_stt)
{
  if(my_ip_addr != INADDR_NONE)
    return my_ip_addr;
  else {
#if defined(__linux) || defined(__linux__) || defined(linux)
    *self_stt = 0;
#else
    const char *host_name;
    const struct  hostent *hostptr;
#ifdef _WIN32
    char name_buf[256]="";
    DWORD name_len=sizeof(name_buf);

    if(!GetComputerName(name_buf, &name_len)) {
      DWORD err;
      err = GetLastError();
      FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, NULL, err, 0, name_buf, sizeof(name_buf),NULL);
      printf("get_frst_self_ip(): GetComputerName() failed Err=%d (%s)\n",
             (int)err, name_buf);
      *self_stt = NULL;
      return INADDR_NONE;
    }
    host_name = name_buf;
#else           
    struct  utsname uts;

    uname(&uts);
    host_name = uts.nodename;
#endif

    /* retreive node name */
    hostptr = gethostbyname(host_name);
    if(hostptr == NULL) {
      const char *err_str =
        h_errno==HOST_NOT_FOUND? "HOST_NOT_FOUND":
        h_errno==NO_ADDRESS? "NO_ADDRESS":
        h_errno==NO_DATA? "NO_DATA":
        h_errno==NO_RECOVERY? "NO_RECOVERY":
        h_errno==TRY_AGAIN? "TRY_AGAIN": "Unknown";
                        
      printf("get_frst_self_ip(): gethostbyname(\"%s\")failed. err=%d (%s)\n",
             host_name, h_errno, err_str);
#ifndef _WIN32                                  /* In _WIN32 this is not an L-value */
      h_errno = 0;
#endif
      *self_stt = NULL;
      return INADDR_NONE;
    }
    *self_stt = (char **)hostptr->h_addr_list;
#endif
  }
  return get_next_self_ip(self_stt);
}

in_addr_t get_next_self_ip(get_self_ip_t *self_stt) {
  in_addr_t ip;

  if(my_ip_addr != INADDR_NONE)
    return INADDR_NONE;
  else {
#if defined(__linux) || defined(__linux__) || defined(linux)
    ip = get_ifcn_ip(*self_stt);
    *self_stt += 1;
#else
    char **addrptr = *self_stt;

    if(addrptr == NULL || *addrptr == NULL)
      ip = INADDR_NONE;
    else {
      memcpy(&ip, *addrptr++, sizeof(ip));
      ip = ntohl(ip);
      *self_stt = addrptr;
    }
#endif
  }

  return ip;
}

int
upoll(pollfd_t *fds,
      nfds_t nfds,
      unsigned long timeout,    
      int *err,
      const char **errstr
      )
{
  int ret_val;

  /* Having a non zero definition of POLLRDNOHUP seems to indicate that
     ppoll() is available as well */
#if !defined(_WIN32) && !defined(__CYGWIN__) && !defined(__sun__)  && POLLRDHUP
  struct timespec wait, *pwait;

  if(timeout == UPOLL_INDEFINITE) {
    pwait = NULL;
  } else {
    wait.tv_sec = timeout/1000000;
    wait.tv_nsec = (timeout%1000000)*1000;
    pwait = &wait;
  }
  ret_val = ppoll(fds, nfds, pwait, NULL);

#else  /* !defined(_WIN32) && !defined(__CYGWIN__)  && !defined(__sun__)  && POLLRDHUP*/

#ifdef _WIN32
#define USE_SELECT 1                    /* Must be - Windows XP does not have poll() */
#else
#define USE_SELECT 0
#endif /* definition of USE_SELECT */

  int max_sock = -1;
  nfds_t k;
  pollfd_t *pfd;
#ifdef _WIN32
  pollfd_t alt_fd;
  int active_sock = 0;
#endif /* def _WIN32 */
#if USE_SELECT
  fd_set rset, wset, eset, *prset=NULL, *pwset=NULL, *peset=NULL;
  struct timeval wait, *pwait;

  FD_ZERO(&rset); FD_ZERO(&wset); FD_ZERO(&eset);
#else
  int wait;
#endif /* USE_SELECT */

  INIT_IP_STACK();

#ifdef _WIN32
  alt_fd.fd = upoll_dummy_sock;
  alt_fd.events = POLLFAIL;
  alt_fd.revents = POLLFAIL;
#endif /* def _WIN32 */

  for(k=0; k<nfds; k++) {
    pfd = &fds[k];

    if(!pfd->events) {
#ifdef _WIN32
      pfd->revents = 0;
#endif /* _WIN32 */
      continue;
    }
#if USE_SELECT
    if(pfd->events & POLLREAD) {
      prset = &rset;
      FD_SET(pfd->fd, prset);
#ifdef _WIN32
      active_sock = 1;
#endif /* def _WIN32 */
    }
    if(pfd->events & POLLOUT) {
      pwset = &wset;
      FD_SET(pfd->fd, pwset);
#ifdef _WIN32
      active_sock = 1;
#endif /* def _WIN32 */
    }
    if(pfd->events & POLLFAIL) {
      peset = &eset;
      FD_SET(pfd->fd, peset);
#ifdef _WIN32
      active_sock = 1;
#endif /* def _WIN32 */
    }
#ifndef _WIN32
    if(pfd->fd > max_sock)
      max_sock = pfd->fd;
#endif
#endif /* USE_SELECT */

  } /* for(k=0; k<nfds; k++)  */

#ifdef _WIN32
  if(!active_sock) {                    /* No socket - create a dummy socket */
    fds = &alt_fd;
    peset = &eset;
    FD_SET(upoll_dummy_sock, peset);
  }
#else  /* _WIN32 */

  max_sock += 1;

#endif /* def _WIN32 */

#if USE_SELECT
  if(timeout == UPOLL_INDEFINITE) {
    pwait = NULL;
  } else {
    wait.tv_sec = timeout/1000000;
    wait.tv_usec = (timeout%1000000);
    pwait = &wait;
  }

  ret_val = select(max_sock, &rset, &wset, &eset, pwait);
  if(ret_val >= 0) {
#ifdef _WIN32
    if(fds == &alt_fd)
      ret_val = 0;
    else
#endif /* def _WIN32 */
      for(k=0; k<nfds; k++) {
        pfd = &fds[k];
                                
        pfd->revents = 0;
        if(FD_ISSET(pfd->fd, &rset))
          pfd->revents = (short)(pfd->revents | POLLREAD);
        if(FD_ISSET(pfd->fd, &wset))
          pfd->revents = (short)(pfd->revents | POLLOUT);
        if(FD_ISSET(pfd->fd, &wset))
          pfd->revents = (short)(pfd->revents | POLLFAIL);
      }
  }
#else  /* USE_SELECT */
  if(timeout == UPOLL_INDEFINITE)
    wait = -1;
  else if(timeout > ULONG_MAX - 999)
    wait = INT_MAX;
  else {
    timeout = (timeout + 999) / 1000;
    wait = (timeout > INT_MAX)? INT_MAX: (int)timeout;
  }

  ret_val = poll(fds, nfds, wait);

#endif /* else USE_SELECT */

#undef USE_SELECT
        
#endif /* else  !defined(_WIN32) && !defined(__CYGWIN__) */

  if(ret_val < 0)
    get_sock_error(err,errstr);

  return ret_val;
}

int
upoll_nointr(pollfd_t *fds,
             nfds_t nfds,
             unsigned long timeout,     
             int *err,
             const char **errstr
             )
{
  int ret_val;
  int error;
  struct timeval begin, now;
  int intr=0;

  if(timeout != UPOLL_INDEFINITE)
    gettimeofday(&begin,NULL);

  for(;;) {
    ret_val = upoll(fds, nfds, timeout, &error, errstr);

    if(ret_val >= 0) break;

    get_sock_error(&error,errstr);
    if(err != NULL) *err = error;

#ifdef __unix
    if(error == EINTR)
      intr = 1;
#elif defined(_MSC_VER)
    if(error == WSAEINTR)
      intr = 1;
#endif
    if(intr) {
      if(timeout != UPOLL_INDEFINITE) {
        unsigned long diff;
                
        gettimeofday(&now, NULL);
        diff = timersub_usec(&now,&begin);
                
        if(diff >= timeout) break;
                
        begin = now;
        timeout -= diff;
      }
      continue;
    }

    break;
  }

  return ret_val;
}
