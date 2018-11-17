#ifdef _MSC_VER
#define F_DECLARE extern "C" __declspec(dllimport)
#else
#define F_DECLARE __attribute__((__visibility__("default")))
typedef void *HANDLE;
extern F_DECLARE int _getch(void);
extern F_DECLARE int _kbhit(void);
#include <stdio.h>
#include "win2nix.h"
#endif
#include "SAVparams.h"

/* friendlier name for skin for user apps */
#define SKIN_COLOR SKIN_TABLE_VALUE

/* length of SAVavg file header */
#define SAVAVG_FILE_HEADER_LEN (HEADER_LENGTH+5)

/* heat map type */
#define VIDEO_FRAME 0
#define DENSITY_MAP 1
#define DIRECTION_MAP 2
#define VELOCITY_MAP 3
#define COLOR_MAP 4
#define FAST_PLOT 5

typedef struct SAVlevel SAVlevel_t;
typedef struct {int argc; char **argv;} Arg_t;

#ifndef NO_API_PROTOTYPES
#ifndef NO_OPENCV

F_DECLARE
long _vaatMotFeats(HANDLE);

F_DECLARE
SAVlevel_t *_vaatSAVbar(HANDLE, int *nBlkNot0);

F_DECLARE
int _vaatTrendPrlUpdate(HANDLE);

F_DECLARE
int _vaatTrendPrlUpdateExplicit(HANDLE, int);

F_DECLARE
void _vaatSetTrendTime(HANDLE, time_t t);

F_DECLARE
int _vaatTrendEnvel(HANDLE handle);

F_DECLARE
int _vaatTrendDiff(HANDLE);

F_DECLARE
CvCapture *_vaatGetVideoIn(HANDLE);

F_DECLARE
IplImage *_vaatGetCurrentVideoFrame(HANDLE);

F_DECLARE
void _vaatStoreThruCurrentVideoFrameAddr(HANDLE, IplImage *);

F_DECLARE
IplImage *_vaatGetCurrentVideoBG(HANDLE);

F_DECLARE
IplImage **_vaatGetCurrentVideoFrameAddr(HANDLE);

F_DECLARE
VidParam_t *_vaatGetVideoCaps(char *cameraSpec);
#endif /* NO_OPENCV */

F_DECLARE
void _vaatDisplayHelp(int argc, char *argv[], const char *text[]);

F_DECLARE
char * _vaatGetVideoDescription(HANDLE handle);

F_DECLARE
HANDLE _vaatInitialize(char *args[], int *errCode);

F_DECLARE
HANDLE _vaatInitializeEx(char *args[], int *errCode);

F_DECLARE
const HANDLE _vaatStart(VidParam_t *p, int levelType, char *trendDir);

F_DECLARE
const HANDLE _vaatGetHandle(void);

F_DECLARE
void _vaatSetDimSS(HANDLE, long widSS, long htSS);

F_DECLARE
void _vaatClearSAV(HANDLE, int levelType);

F_DECLARE
void _vaatTerminate(HANDLE);

F_DECLARE
void _vaatDisplay(HANDLE, int wantPrivacy, short *isInitialized);

F_DECLARE
int _vaatGetSAVbyLevel(HANDLE handle, int iFeatsLevel, int levelType, struct FeatsList *featsList, int featsLen, int densThresh);

F_DECLARE
int _vaatGetType(HANDLE);

F_DECLARE
VidParam_t *_vaatGetVideoParms(HANDLE);

F_DECLARE
long _vaatGetDisplayLevel(HANDLE);

F_DECLARE
short _vaatGetSetBgFlag(HANDLE);

F_DECLARE
void _vaatSetSAV(HANDLE, SAVlevel_t *l);

F_DECLARE
void _vaatSetSAVlevels(HANDLE, long level);

F_DECLARE
void _vaatSetSetBgFlag(HANDLE, short flag);

F_DECLARE
long _vaatGetFgPct1000(HANDLE);

F_DECLARE
int _vaatGetVideoWidth(HANDLE);

F_DECLARE
int _vaatGetVideoHeight(HANDLE);

F_DECLARE
int _vaatGetFps(HANDLE);

F_DECLARE
int _vaatGetFeatBlkCnt(SAVlevel_t *l, int nLevel);

F_DECLARE
SAVlevel_t *_vaatGetSAVbyType(HANDLE, int levelType);

F_DECLARE
int _vaatUnpackSAVbyType(HANDLE handle, int levelType, time_t *ts, int size, unsigned char *buf);

F_DECLARE
int _vaatGetNumSAVlevels(HANDLE);

F_DECLARE
long _vaatGetNumFramesPerSAVbar(HANDLE);

F_DECLARE
void _vaatCheckBgReset(HANDLE handle, VidParam_t *p, short *manualBgReset);

F_DECLARE
void _vaatSetNumFramesPerSAVbar(HANDLE, int value);

F_DECLARE
int _vaatCreateSAV(struct SAVlevel **level, long widSS, long htSS);

F_DECLARE
void _vaatZeroSAV(struct SAVlevel *level, long nLevels);

F_DECLARE
int _vaatPackSAV(HANDLE, time_t ts, int nLevels, SAVlevel_t *level);

F_DECLARE
int _vaatUnpackSAV(HANDLE, time_t *ts, int nLevels, SAVlevel_t *level);

F_DECLARE
const u_char *_vaatGetPackedSAV(HANDLE);

F_DECLARE
int _vaatGetPackedSAVsize(HANDLE);

F_DECLARE
const char *_vaatGetPackerErrMsg(HANDLE);

F_DECLARE
const char *_vaatInitSAVstore(HANDLE, const char *prefix, int period, int retention, __int64 maxSize);

F_DECLARE
void _vaatWriteSAV(HANDLE, const u_char *buf, u_short len);

F_DECLARE
void _vaatStoreSAVbar(HANDLE);

F_DECLARE
int _vaatSAVfileOpen(HANDLE handle, char *fName);

F_DECLARE
int _vaatSAVfilePrepare(HANDLE handle, FILE *fp);

F_DECLARE
int _vaatSAVfileUse(HANDLE handle, char *fName);

F_DECLARE
SAVlevel_t *_vaatGetSAV(HANDLE handle, time_t *ts, int *levels, int *packedSize, int *nVframes);

F_DECLARE
int _vaatGetFrameRate(VidParam_t *parms);

F_DECLARE
int _vaatGetConfigInt(HANDLE handle, const char *configKey);

F_DECLARE
double _vaatGetConfigDbl(HANDLE handle, const char *configKey);

F_DECLARE
const char *_vaatGetConfigStr(HANDLE handle, const char *configKey);

F_DECLARE
SAVfileAttr_t *_vaatGetSAVfileAttr(HANDLE handle);

F_DECLARE
SAVpackRslt_t *_vaatSerializeSAV(HANDLE handle, time_t time, int nlevel, const struct SAVlevel *level, int style);

F_DECLARE
int _vaatGetPackerStyle(HANDLE handle, int isWideData, int isPruned);

F_DECLARE
const unsigned char *_vaatGetRawSAVavgFileHeader(HANDLE handle, const SAVfileAttr_t *sfa);

F_DECLARE
const int _vaatGetSSfactor();

F_DECLARE
int _vaatGetPredominantFeature(HANDLE handle, int iLevelFeats, int featNum, int levelType, int densThresh);

F_DECLARE
IplImage *_vaatRenderFastPlot(HANDLE handle);

F_DECLARE
void _vaatBuildFastPlot(HANDLE handle);

#ifdef _MSC_VER
F_DECLARE
HBITMAP _vaatGetMapAsHBitMap(HANDLE handle, int mapType);

F_DECLARE
HBITMAP _vaatGetHbitMap(HANDLE handle, int mapType);
#endif

F_DECLARE
int _vaatGetPredominantFeatureBySAV(int iLevelFeats, int featNum, SAVlevel *level, int densThresh);

F_DECLARE
int _vaatListLevel(struct SAVlevel *level, int iFeatsLevel, struct FeatsList *featsArray, int featsLen, int densThresh);

F_DECLARE
void _vaatRescaleSAV(SAVlevel *level, int nLevels);

F_DECLARE
int _vaatCountArgs(char *argv[]);

F_DECLARE
Arg_t extractArgs(int *argc, char *argv[], const char *spec);
#endif /* NO_API_PROTOTYPES */
