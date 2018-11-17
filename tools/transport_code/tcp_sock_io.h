/** \file tcp_sock_io.h defintions and prototypes for TCP sockets communication */

#ifndef TCP_SOCK_IO_H
#define TCP_SOCK_IO_H

#if defined(__unix)
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#ifndef SOCKET_T_DEFINED
#define SOCKET_T_DEFINED
typedef int socket_t;
#endif

#define SOCKET_T_UNDEF ((socket_t)-1)
#define SOCKET_ERROR (-1)

#ifndef SOCKLEN_T_DEFINED
typedef socklen_t SOCKLEN_T;
#define SOCKLEN_T_DEFINED
#endif

#ifndef SSIZE_T_DEFINED
#define SSIZE_T_DEFINED
typedef ssize_t SSIZE_T;
#endif

#define INIT_IP_STACK()
#define CLOSESOCKET(s) close(s)

#endif	/* #if defined(__unix) */

#if defined(_WIN32) || defined(__MINGW64__)
#include <Winsock2.h>
#include <Ws2tcpip.h>                   /* See MSDN Article ID: 257460 */
typedef unsigned long in_addr_t;
typedef unsigned short in_port_t;

#ifndef SOCKET_T_DEFINED
#define SOCKET_T_DEFINED
typedef SOCKET socket_t;
#endif

#define SOCKET_T_UNDEF (INVALID_SOCKET)

#ifndef SOCKLEN_T_DEFINED
typedef int SOCKLEN_T;
#define SOCKLEN_T_DEFINED
#endif

#if !defined(SSIZE_T_DEFINED) && !defined(_BASETSD_H_)
#define SSIZE_T_DEFINED
typedef long SSIZE_T;
#endif

void init_winsock(void);
#define INIT_IP_STACK() init_winsock()
#define CLOSESOCKET(s) closesocket(s)

/** Windows replacement to snprintf */
#ifndef snprintf
#define snprintf sprintf_s
#endif

#endif  /* _WIN32 */

/* Some OS (like sun) don't have SOL_IP defined - they use the protocol family number */
#ifndef SOL_IP
#define SOL_IP IPPROTO_IP
#endif
#ifndef SOL_IPV6
#define SOL_IPV6 IPPROTO_IPV6
#endif

/* Some definitions for upoll() */
#if !defined(_WIN32) && !defined(__MINGW64__)
#include <poll.h>

#ifndef __SUNPRO_C                              /* In Sun native C compiler pollfd_t is defined in poll.h */
typedef struct pollfd pollfd_t;
#endif

#else  /* _WIN32 case */

typedef struct pollfd pollfd_t;

typedef int nfds_t;
#endif


/* POLLRDHUP is not defined everywhere.  Add this execption for every architecture where needed. */
#ifndef POLLRDHUP
#define POLLRDHUP 0
#endif

#define POLLFAIL  (POLLERR | POLLHUP | POLLNVAL | POLLRDHUP)
#define POLLREAD  (POLLIN | POLLPRI)
#define POLLWRITE POLLOUT
#define UPOLL_INDEFINITE  ULONG_MAX

#ifndef INADDR_NONE
#define INADDR_NONE INADDR_BROADCAST
#endif

#ifndef EXT_C
#ifdef __cplusplus
#define EXT_C extern "C"
#else
#define EXT_C
#endif
#endif

#define IP_FMT "%lu.%lu.%lu.%lu"
#define IP_PRNT(ip) ((unsigned long)(ip))>>24,(((unsigned long)(ip))>>16)&0xFF, \
    (((unsigned long)(ip))>>8)&0xFF,((unsigned long)(ip))&0xFF

struct sockaddr;

EXT_C void
get_sock_error(int *err,	/* If not NULL, returns error code */
               const char **err_str /* It not NULL returns error
				       string, or NULL if successful */
               );

EXT_C void
ip_to_str(in_addr_t ip,         /* host order */
          char *str
          );

/** Read an IP address represented by \c str. 
    \c str may contain the actual IP address, either as 0{x|X}\<hex no\> or in
    a dotted decimal notation (x.y.u.z) If \c str contains a column (:), 
    the last column in the string and anything following it are ignored.
    \param str is host name string.
    \param next returns a pointer to the end of the decoded address string
    (if not NULL). Thus, if decoding failed it returns a pointer to str; else
    if str contains a column it returns a pointer to the final column; else it
    returns a pointer to the terminating null character.
    \return the IP address in host
    order or INADDR_NONE if the host was not found.
*/
EXT_C in_addr_t
str_to_ip(const char *str,
          char **next
          );                                            /* IP returned in host order */

/** Read an IP address represented by \c str. 
    \c str may contain the actual IP address, either as 0{x|X}\<hex no\> or in
    a dotted decimal notation (x.y.u.z).  Alternatively, \c str may contain the host
    name. In either case, if \c str contains a column (:), the last column in
    the string and anything following it are ignored.
    \param str is host name string.
    \param next if not NULL, returns a pointer to the end of the decoded
    address string. Thus, if decoding failed it returns a pointer to str; else
    if str contains a column it returns a pointer to the final column; else it
    returns a pointer to the terminating null character.
    \return the IP address in host
    order or INADDR_NONE if the host was not found.
*/
EXT_C in_addr_t
name_to_ip(const char *str,
	   char **next
	   );

/* Force a self IP address. */
EXT_C void set_my_ip_addr(in_addr_t ip);

#if defined(__linux) || defined(__linux__) || defined(linux)

/** Get IP address of an interface
    \return the IP address (host order) corresponding to an
    interface. If no IP is available INADDR_NONE is returned */
EXT_C in_addr_t get_ifc_ip(const char *ifc);

/* get_ifcn_ip() returns the IP address (host order) corresponding to the
   interface "ethN", where N is an integer argument. If no IP is available
   INADDR_NONE is returned */

EXT_C in_addr_t get_ifcn_ip(unsigned num);

#endif /* defined(__linux) || defined(__linux__) || defined(linux) */

/* The functions get_frst_self_ip() and get_next_self_ip() are used to loop
   over the IP addresses of the host. They use a variable of of type
   get_self_ip_t to keep state information.  They return an IP address in
   host order or INADDR_NONE if no more IPs are available.  Typical usage is:

   in_addr_t ip;
   get_self_ip_t stt;
   for(ip=get_frst_self_ip(&stt); ip!=INADDR_NONE; ip=get_next_self_ip(&stt))
   ... Do something with ip
   
*/

#if defined(__linux) || defined(__linux__) || defined(linux)
typedef unsigned get_self_ip_t;
#define UNDEF_SELF_IP  ((get_self_ip_t)0)
#else
typedef char ** get_self_ip_t;
#define UNDEF_SELF_IP  ((get_self_ip_t) NULL)
#endif

EXT_C in_addr_t get_frst_self_ip(get_self_ip_t *self_stt);
EXT_C in_addr_t get_next_self_ip(get_self_ip_t *self_stt);

/** A hybrid between poll() and ppoll(). No sigmask and time is specified as
    an integer, but the integer is unsigned long, has usec resolution and a
    value of UPOLL_INDEFINITE means block indefinitely.
    \param fds array of \c nfds sockets to monitor.
    \param nfds no. of elements in \c fds.
    \param timeout  timeout in usec.  0 = return immediately, \c
    UPOLL_INDEFINITE = block indefinitely.
    \param err If not NULL and return value is -1 returns error code.
    \param errstr If not NULL and return value is -1 returns error string.
    \return no. of revent members which are not 0. 0 indicates timeout. -1=error.
*/
EXT_C int
upoll(pollfd_t *fds,
      nfds_t nfds,
      unsigned long timeout,    
      int *err,
      const char **errstr
      );

/* Same as upoll(), but if an interrupt causes upoll/select to exit, will keep polling */
EXT_C int
upoll_nointr(pollfd_t *fds,
             nfds_t nfds,
             unsigned long timeout,     
             int *err,
             const char **errstr
             );

/** Open a TCP socket in server (listening) mode and wait for one
    client to connect

    \param lcl_host - Local interface to bind to. Can be a name or a
      dotted decimal notation of an IP address. If NULL, INADDR_ANY
      is used.
    \param lcl_port - Local port to use.
    \param hostlen - size available for returning cliet IP address.
    \param host - (output) a pointer to NULL or to an char array of
      size \c hostlen. Returned the IP address of the client in
      dotted decimal notation.
    \param port - (output) the port number (host order)of the client.
    \param timeout - duration (sec.) to wait for connection. Wait indefinitely
                     if \c timeout == 0.
    \param err_str - (output) NULL - no error. Otherwise and error
      string.

    \retun a socket_t object opened according to the
      arguments.connected to the client. The listening socket is closed.
    
*/
EXT_C socket_t 
openTCPSocketServer(const char *lcl_host, 
		    unsigned long lcl_port,
		    size_t hostlen,
		    char *host,
		    unsigned long *port,
		    double timeout,
		    const char **err_str
		    );

/** Open a TCP socket in client mode and connect to a server

    \param host - Name or dotted decimal notation of server address.
    \param port - port number (host order) on the server.
    \param lcl_host - Local interface to bind to. If NULL, INADDR_ANY
      is used.
    \param lcl_ port Local port to
      use. Normally should be 0 to let the system choose one.
    \param err_str - output: NULL - no error. Otherwise and error
      string.

    \retun a socket_t object opened according to the arguments.
    
*/
EXT_C socket_t 
openTCPSocketClient(const char *host,
		    unsigned long port,
		    const char *lcl_host, 
		    unsigned long lcl_port,
		    const char **err_str
		    );

/** Send data of length datalen over the socket. \return error code or
    zero for success */
EXT_C int sendTCPSocket(socket_t sock,
			size_t datalen,
			const void *data,
			const char **err_str /**< output: NULL - no
						error. Otherwise and error string. */
			);

/** Receive datalen bytes from the socket.
    \return 
    - if successful, number of bytes which were actually read
    - SOCKET_ERROR for error or remote side gracefuly closed
    - 0 for time out (or if requested to read 0 bytes).
    One can distinguish between an error and graceful closure by checking *err_msg.
    err_msg points to an error code if an error occurred or is NULL otherwise.
    */
EXT_C SSIZE_T
recvTCPSocket(socket_t sock,
	      size_t datalen,	/**< number of bytes to read */
	      void *data,	/**< output: returned data*/
	      double timeout,	/**< duration to wait (sec.) for data. 0=wait indefinitely */
	      const char **err_str /**< output: NULL - no error. Otherwise an error string. */
	      );

/** Close a socket. \return 0 for success or an error code */
EXT_C int closeTCPSocket(socket_t sock,
			 int timeout, /**< control SO_LINGER. -1 l_onoff=0 (default)
				         >=0, l_onoff=1 and linger is the value of l_linger */
			 const char **err_str
			 );


#endif	/* TCP_SOCK_IO_H */
