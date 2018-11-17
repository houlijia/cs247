#ifndef __TCP_SOCKET_H__
#define __TCP_SOCKET_H__

#ifdef __cplusplus
extern "C" {
#endif
void DieWithError(const char *errorMessage);  /* Error handling function */

int AcceptTCPConnection(int servSock) ;

int CreateTCPServerSocket(unsigned short port) ;

void HandleTCPClient(int clntSocket) ;

#ifdef __cplusplus
}
#endif

#endif 
