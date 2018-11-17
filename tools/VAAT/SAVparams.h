/* SAVPARAMS.H - parameter values for SAV functions
 */

#ifndef _SAVPARAMS_H // take out this comment
#define _SAVPARAMS_H
#include <highgui.h>

typedef unsigned char u_char;
typedef unsigned short u_shrt;

#define NDIRN 9

/* color */
#define COLORBAND_RGB 0
#define COLORBAND_SKIN 1
#define SKIN_BLOCK_THRESH_PCT 30			// if greater than this percentage skin pixels in block, then block set to skin color
#define SKIN_TABLE_VALUE 64						// color quantization r,g,b (2 bits each), leaving 2 bits of which value=64 -> skin

/* Fast plot */
#define N_FAST_DATA 200
struct RecentData{
	int current, trend;
};

#define DISPLAY_LEVEL_DFLT 4	// SAV scale that is displayed (can be changed during runtime)

/* structures for SAV */
#define FEAT_SCALE SHRT_MAX
#define FEAT_SCALE_F (double)SHRT_MAX
#define MIN_SIDE 3						// smallest sidelength of multi-scale blocks

/* 3 SAV options calculation and storage options */
#define LEVELTYPE_BAR 0					// SAV averaged over SAVbar interval
#define LEVELTYPE_TREND_LIN 1		// running average of SAVbar samples over linear time
#define LEVELTYPE_TREND_PRL 2		// running average of SAVbar samples over parallel time (e.g., 2pm each day) from file 
#define LEVELTYPE_TREND_BOTH 3	// 2 running avgs of SAVbar samples, linear and parallel

/* define how we will log SAVs */
enum {
	NARROW_DATA_AND_PRUNE,
	NARROW_DATA_NO_PRUNE,
	WIDE_DATA_NO_PRUNE
};
#define PACK_STYLE NARROW_DATA_NO_PRUNE

/* FeatsList structure takes feature values from SAV structure to a simple list
   and includes feature values for all blocks of a chosen level */
#ifdef _MSC_VER
#pragma pack(push)
#pragma pack(1)
#endif
struct FeatsList{
	u_shrt iXBlk, iYBlk;											// indices of block
	u_shrt x, y;															// center coord.s of block
	u_shrt density, vel;											// density, velocity values [0-1000]
	u_shrt densDev, dirnDev, velDev, clrDev;	// deviation values [0-1000]
	u_char r, g, b, skin;											// rgb colors [0-3]; skin [0=no; 1=yes]
	u_shrt dirn;															// [0, 1-8]: 0 (no direction); 1-8 (E, SE, S, ..., NE)
#ifdef _MSC_VER
#pragma pack(pop)
};
#else
} __attribute__((packed));
#endif

/* structures for SAV */

/* Block Features - these are used to accumulate over frames before averaging; note dirn,clr are broken out for summation */
struct BlockFeats2
{
	int density;					// scaled *FEAT_SCALE; no. motion pixels per block area
	int densityDev;				// scaled *FEAT_SCALE; Euclidian dist centroid from center divided by block sidelength (shorter if diff)
	int dirn;							// this not used for accumulation - sumCos and sumSin are used for to avg within SAVbar
	int sumCos, sumSin;		// scaled *FEAT_SCALE; x,y increment of direction, so that accum. can be done without circular avg
	int dirnDev;					// scaled *FEAT_SCALE; nonuniformity of direction histogram: 255 (max) is single bin, 0 is uniform distr
	int vel;							// scaled *FEAT_SCALE; velocity:  0=static, FEAT_SCALE=(fastest) (relative to pix res, not real world) 
	int velDev;						// *FEAT_SCALE; velocity deviation
	int clr;							// 2-2-2+2 RGB quantization r,g,b (2 each) + 1 (skin) +1 (other); clr not used for accum; clr* are
	int clrDev;						// *FEAT_SCALE; color deviation
};

struct Block2
{
	long x, y;
	struct BlockFeats2 blockFeats;
	long nBlkMotion;			// tallies the number of blocks that have some motion (0-motion blocks are not averaged)
};

struct SAVlevel
{
	long iLevel;								// index of a level
	long widBlock;							// width of blocks on a level
	long htBlock;								// height of blocks on a level
	long nXBlock, nYBlock;			// number of blocks in x and y axes on a level
	struct Block2 *block;				// block structure containing feats for each block on a level
};

struct Level2									// NOTE - same as SAVlevel; deprecate this after SAVStore2 is removed (log 21-Jan-14)
{
	long iLevel;								// index of a level
	long widBlock;							// width of blocks on a level
	long htBlock;								// height of blocks on a level
	long nXBlock, nYBlock;			// number of blocks in x and y axes on a level
	struct Block2 *block;				// block structure containing feats for each block on a level
};

/* trend structure */
struct Trend{
	long tFrom0;								// number of seconds from time 0 for day; [log 16Sep14] not used now; I forget purpose...
	long nLevels;								// number of SAV levels
	long wid, ht;								// size of original image
	long tWndw;									// time window within which multiple SAVbar samples are combined to 1 SAVtrend sample
	long nAccum;								// accumulating number of trend samples
	long densRecent;						// change in trend density block 0 for recent samples according to time constant
	struct SAVlevel *level;			// SAV level structure for trend update
};

/* IP Camera characteristics */
#define MAX_IPCAMS 20					// max number of IP camera
struct IPcam
{
	char name[64];							// name of location of camera
	char login[128];						// login to ip cam
	char ipAddress[MAX_IPCAMS];	// IP address of camera
	char port[16]; 							// the port number, for socket initialization, default is IP_PORT_DFLT
	char ipCamType[128];				// for different init strings
};

typedef struct {
#ifndef NO_OPENCV
	CvCapture* videoIn;
	char vidFileNameIn[256];
	struct IPcam ipCam;
	int camID;
	int fps;
	int widVideo;
	int htVideo;
#endif
	int sizeVideo;
} VidParam_t;

/* SAVavg file characteristics */
typedef struct {
	int widSS;
	int htSS;
	int isWideData;
	int isPruned;
} SAVfileAttr_t;

/* SAV packing results */
typedef struct {
	int len;
	const unsigned char *data;
} SAVpackRslt_t;

#endif	// _SAVPARAMS_H