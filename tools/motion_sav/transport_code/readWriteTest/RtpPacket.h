#ifndef RTP_PACKET_H
#define RTP_PACKET_H
#include "common_cpp.h"
#include "common_ljw.h"

#include <string>
#include <cstdio>
#include <cassert>


const uint32 MAX_RTP_DATA_LEN = 100000;
/**
 * @class This is the class that representing a RTP packet
 * @param timeStamp
 * @param seq
 * @param payloadType
 * @param ssrc
 * @param dataSrc
 * @param dataLen0
 */
class RtpPacket
{
public:
  /**
   * @breif contructor
   * @param timeStamp -The time stamp of the packet
   * @param seq -The sequence number
   * @param payloadType -The payload Type
   * @param dataSrc -The source of payload data
   * @param dataLen0 -The length of the data
   */
  RtpPacket (uint32 timeStamp, uint16 seq, char payloadType, uint32 ssrc, char* dataSrc, uint32 dataLen0);
  RtpPacket (char payloadType, uint32 ssrc, uint32 maxLen);

  RtpPacket (FILE* fp);
  RtpPacket (char* buffer, uint32 len);


  char readBasicHeaderFromBuffer (char* buffer, const char** errString);

uint32 readDataFromBuffer (char* buffer, uint32 toReadLen, const char** errString);
  /**
   *@notice Any constructor should set those pointer to NULL first
   */
  ~RtpPacket ()
  {
    if (csrc != NULL)
      {
        free (csrc);
        csrc = NULL;
      }
    if (data != NULL)
      {
        free (data);
        data = NULL;
      }
  }

  /**
   * @brief Utility function to set an int32 into big endian order\
   * @param start -The starting position of the value in the buffer
   * @param value -The value
   */
  inline void
  setBigEInt32 (uint8* start, uint32 value)
  {
    start[3] = value & 0xFF;
    value >>= 8;
    start[2] = value & 0xFF;
    value >>= 8;
    start[1] = value & 0xFF;
    value >>= 8;
    start[0] = value & 0xFF;
  }

  /**
   * @brief Utility function to set an int16 into big endian order
   * @param start -The starting position of the value in the buffer
   * @param value -The value
   */

  inline void
  setBigEInt16 (uint8* start, uint16 value)
  {
    start[1] = value & 0xFF;
    value >>= 8;
    start[0] = value & 0xFF;
  }

  /**
   * @brief Get an 16 bit int value from the buffer
   * @param The start of the buffer
   */

  inline uint
  getBigEInt16 (const uint8* start) const
  {
    return (start[0] << 8) | start[1];
  }

  /**
   * @brief Get the sequence number
   */
  uint16
  getSeqNo () const
  {
    return getBigEInt16 (& (fixedHeader[2]));
  }

  /**
   * @brief Get the lenght of the RTP header
   * we may add the extra length later in this function based on the extension bit
   */
  uint32
  getHeaderLength () const
  {
    return 12;
  }

  /**
   * @brief Get the length of data
   */

  uint32
  getDataLength () const
  {
    return dataLen;
  }
/**
 * @brief return the total length of the rtp packet, including the header
 * @return 
 */
  uint32 getTotalLength() const
  {
   return dataLen +getHeaderLength();
  }

  /**
   * @brief Get the data pointer
   */
  char*
  getData () const
  {
    return data;
  }

  const char*
  getReadOnlyData() const
  {
   return data;
  }

  const char*
  getReadOnlyHeader() const
  {
   return (const char*) fixedHeader;
  }


  /**
   * @brief Write the current rtp packet into a file, the name
   * of the file is just "sequence number"+".rtp".
   * @param folderName -The name of the folder, usually it is the time that the program is run
   * @param errString -Error string
   * @return -The file length written
   */
  uint16 writeToFile (string folderName, uint16 baseNumber, const char**errString) const;

  //void readBasicHeaderFromFile(string folderName, char**errString);

  /**
   * @brief Set the timestamp to a uint32 value
   */
  void
  setTimeStamp (uint32 timeS)
  {

    setBigEInt32 (& (fixedHeader[4]), timeS);
  }

  /**
   * @brief Set the sequence number
   */
  void
  setSeqNo (uint16 seq)
  {

    setBigEInt16 (& (fixedHeader[2]), seq);
  }

  void
  setDataLen (uint32 dataLen0)
  {
    dataLen = dataLen0;
  }

  //friend class UdpRTPSender;

private:
  char readBasicHeader (FILE* fp, const char** errString);

  /**
   * @brief Currently, we assume there is a maxium length of rtp packets/packet files.  I just read MAX_RTP_DATA_LEN bytes, try to read the max, and the return value of fread is just the real length. I think we can use the length in the packetLen.txt if we really want to know the length before reading. But, I want to make those function independent of that txt file. In later UDP transimission, we can use the UDP payload length to easily get the RTP length. But, I have reserved the toReadLen, and the return value. In case we want to change it, the function interface can remain the same.
   * @param fp -File pointer to read from
   * @param toReadLen -Not used right now. if set to 0 not used, if set to a 
   * positive number, use it
   * @param errString -error string
   *
   */
  uint32 readData (FILE* fp, uint32 toReadLen, const char** errString);

  /**
   * @brief Set the payload type field
   * maintain the marker bit. Since loadType only has 7 bits, so
   * I used char as a paremeter, and assert it to be >=0
   */
  void
  setPayLoadType (char loadType)
  {
    assert (loadType >= 0);
    fixedHeader[1] &= 0x80;
    fixedHeader[1] |= loadType;
  }

  /**
   * @brief Set the SSRC field
   */
  void
  setSSRC (uint32 ssrc)
  {
    setBigEInt32 (& (fixedHeader[8]), ssrc);
  }

  void
  setVersion ()
  {
    fixedHeader[0] |= 0x80;
  }


  //for header storage, used an uint32 array because of the ease to write
  uint8 fixedHeader[12];
  uint8* csrc;
  //the dataLen, the total file length should be rtpp.getHeaderLength() + rtpp.getDataLen();
  uint32 dataLen;
  char* data;

};

#endif
