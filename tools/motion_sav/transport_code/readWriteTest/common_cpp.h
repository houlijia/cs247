#ifndef COMMON_LJW_CPP_H
#define COMMON_LJW_CPP_H 

#include <string>
#include <cstring>
#include <cstdio>

#include "common_ljw.h"
/**
 * @brief This is a utility function to covert a uint32
 * into a string
 * @param number -The uint32 to be converted
 * @return The output string
 */
using std::string;

inline string
convertInt (uint32 number)
{
  if (number == 0)
    return "0";
  string temp = "";
  string returnvalue = "";
  while (number > 0)
    {
      temp += number % 10 + 48;
      number /= 10;
    }
  for (uint i = 0; i < temp.length (); i++)
    returnvalue += temp[temp.length () - i - 1];

  return returnvalue;
}

inline string intToString(uint32 number, char width)
{
 char* buffer = (char*) malloc(width);
 sprintf(buffer, "%05d", number);
 string re(buffer);
 delete buffer;
 return re;
}

#endif
