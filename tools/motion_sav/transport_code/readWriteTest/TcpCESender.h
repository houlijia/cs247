#include "RtpPacket.h"
//#include "UniformQuantizer.h"
#include "Folder.h"
#include "writeUInt.h"

//#define READ_FILE_BUFFER_SIZE 1024

using std::cout;
using std::cin;
using std::endl;


class TcpCESender
{
public:
  TcpCESender(const string& ip, const string& port)
{
   buffer = (char *) malloc (READ_FILE_BUFFER_SIZE * sizeof (char));
  if(buffer == NULL)
    {
      *errString = "malloc in readFile failed";
      return;
    }
}
~TcpCESender()
{

}
  int send();

private:
  FILE* rfp; ///< The file to read from
  char* buffer; 
  const int READ_FILE_BUFFER_SIZE = 1024;
};

