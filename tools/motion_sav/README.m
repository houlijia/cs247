% Yen-Ming Lai
% Alcatel-Lucent Bell Labs summer intern
% Applied Mathematics
% University of Maryland, College Park
% ylai@amsc.umd.edu
%
% 8/12/2011
%
% [Code Handoff README]
%
% The main object developed over the summer is the CompressedSensedVideoCodec
%
% type "doc CompressedSensedVideoCodec" to get documentation of all the
% methods. Follow the links of the methods name to get further comments
% (summary,input,output,etc.)
%
% Specific methods of interest are:
%
% doSimulation - runs a series of test (CS video encode + CS video decode), should be rewritten so that __all__ the input values are read
% in as text file rather than hardcoded .  Output is written to a unique named subdirectory every time this method is run. See the method's comments for further details.
%
% run - this method is main work horse.  It does CS video encoding and
% decoding for a specific. It saves the output of the method to file in
% both binary and text format. In addition, it saves the processed video to
% file also. 
%
% [Examples]
%
% 1) to do a full simulation
%
% type "CompressedSensedVideoCodec.doSimulation(0)"
%
% 2) to do a shorter simulation (reading in only a few frames)
%
% type ""CompressedSensedVideoCodec.doSimulation(1)"
%

