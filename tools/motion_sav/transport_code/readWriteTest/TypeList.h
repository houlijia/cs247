#include <vector>
#include "CodeElement.h"
#include <cstdio>
#include <string>
//#include <iostream>

using std::string;
using std::vector;

class TypeList : public CodeElement
{
public:
  TypeList (const CodeElement& ce);

  //	static const uint32 INIT_TYPELIST_SIZE =10;

  ///@brief debuging function

  void
  dumpTypes ()
  {
    printf ("dumping Types:\n");
    for (uint i = 0; i < list.size (); i++)
      //std::cout<<i << "  "<<list[i]<<std::endl;
      printf ("%d %s", i, list[i].c_str ());
  }
  ///@brief return the type based on key
  ///@param key -The key we want to check

  string
  getTypeFromKey (uint32 key)
  {
    return list[key];
  };
  vector <string> list;
};
