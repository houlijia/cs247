#include "TypeList.h"
#include "readUInt.h"
#include <string.h>
#include <assert.h>

/**
 * @brief This copy constructor will form a TypeList from a CodeElement
 * if the buffer reading as problem, the program will jump out due to the 
 * last assert instrument
 */
TypeList::TypeList (const CodeElement& ce) : CodeElement (ce)
{
  //if user gave a wrong ce, just jump out
  assert (ce.getKey () == 0);

  //if we resize it here, we can not use push_back, has to take care of the index by ourselves
  //I think the default init size of vector is 10, which is large enough for our usage here
  //list.resize(INIT_TYPELIST_SIZE);	
  list.push_back ("Type List");

  int endP;
  const char* errString = NULL;
  char* dataStartP = this->data;
  endP = 0;
  char* bufferEnd = (this->data) + (this->length) - 1;
  while(dataStartP < bufferEnd)
    {
      //read one uint, which is the length of the string
      longInt slen = readOnlyOneUInt (dataStartP, 0, &endP, &errString);
      checkError (errString);
      dataStartP = dataStartP + endP + 1;
      //check the source has enough buffer
      if(dataStartP + slen - 1 > bufferEnd)
        {
          const char* errS = "reading TypeList error: not enough source to read from";
          fputs (errS, stderr);
          exit (0);
        }

      char* str = (char*) malloc (slen + 1);
      memcpy (str, dataStartP, slen);
      str[slen] = '\0';
      //the string copy constructor will have a deep copy, and we are
      //safe to free the string
      list.push_back (string (str));
      free (str);
      dataStartP += slen;
    }

  //this assert make sure we read it corrently
  assert (dataStartP == bufferEnd + 1);
}

