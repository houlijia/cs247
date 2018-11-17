function [nrm_msrs, nrm_prob] = getNormMsrs(inp_list, proc_opts)
  % Reads a list of code sources and returns the normalized
  % measurements (except for the no_clip ones).
  % Input
  %   inp_list - a cell array of file names, or an array of CodeSource
  %              objects from which the measurements are read.
  %   proc_opts (Optional) - a struct with any of the following
  %         fields which impact the processing mode (other fields are
  %         ignored):
  %         prefix - (0ptional) and identifying prefix to add
  %                  before all printing. Default '<Nn>] '
  %         blk_rng - If present specifies blocks range for
  %                   processing. A 2x3 array, upper row is minimum,
  %                   lower row is maximum
  % Output
  %   nrm_msrs - an column vector of the measurements. If the output argument
  %              nrm_prob is specified, this list is sorted in an
  %              increasing order (with repetitions).
  %   nrm_prob - an array of the same size as nrm_msrs. Let N be the
  %              length nrm_prob.  Then nrm_prob(n)=g((n-0.5)/N) where
  %              g() is the inverse of the standard normal
  %              distribution.
  
  if nargin < 2
    proc_opts = struct();
  end
  
  if ~isfield(proc_opts, 'prefix')
    proc_opts.prefix = '';
  end
  nrm_msrs_step = 1000000;
  nrm_msrs_len = 0;
  nrm_msrs = [];
  for k=1:length(inp_list);
    if iscell(inp_list)
      inp = inp_list{k};
    else
      inp = inp_list(k);
    end
    
    prefix_k = sprintf('%s(%d) ', proc_opts.prefix, k);
    fprintf('%sopening %s\n', prefix_k, inp);
    if ischar(inp)
      input = CodeSourceFile(inp);
    else
      input = inp;
    end
    
    dec_info = struct();
    
    while true
      tstart = tic;
      [code_elmnt, ~, dec_info] = ...
        CodeElement.readElement(dec_info, input);
      if ischar(code_elmnt)
        exc = MException('CSVidDecoder:run',...
          ['Error in CodeElement:readElement(): '...
          , code_elmnt]);
        throw(exc);
      elseif isscalar(code_elmnt) && code_elmnt == -1
        break;
      elseif isa(code_elmnt, 'UniformQuantizer')
        dec_info.quantizer = code_elmnt;
        continue;
      elseif isa(code_elmnt, 'VidRegion')
        dec_info.vid_region = code_elmnt;
        if isfield(proc_opts, 'blk_rng')
          rng = proc_opts.blk_rng;
          bx = dec_info.vid_region.blk_indx;
          if ~all(bx <= ones(size(bx,1),1)*rng(2,:))
            break;
          end
        end
        continue;
      elseif isa(code_elmnt, 'CS_EncVidParams')
        dec_info.Yblk_size = code_elmnt.blk_size;
        dec_info.enc_opts = code_elmnt;
        initVBlocker();
      elseif isa(code_elmnt, 'RawVidInfo')
        dec_info.raw_vid = code_elmnt;
        initVBlocker();
      elseif isa(code_elmnt, 'QuantMeasurements')
        if isfield(proc_opts, 'blk_rng')
          rng = proc_opts.blk_rng;
          bx = dec_info.vid_region.blk_indx;
          if ~all(bx >= ones(size(bx,1),1)*rng(1,:))
            continue;
          end
        end
        
        [msrmnts, clipped_indices] = ...
          dec_info.quantizer.unquantize(code_elmnt);
        msrmnts(clipped_indices) = [];
        msrmnts(1:code_elmnt.n_no_clip) = [];
        msrmnts = ...
          (msrmnts - code_elmnt.mean_msr)/code_elmnt.stdv_msr;
        if nrm_msrs_len + length(msrmnts) > length(nrm_msrs);
          new_size = nrm_msrs_step * ...
            ceil((nrm_msrs_len + length(msrmnts)) / nrm_msrs_step);
          tmp = zeros(new_size, 1);
          tmp(1:nrm_msrs_len)= nrm_msrs(1:nrm_msrs_len);
          nrm_msrs = tmp;
        end
        nrm_msrs(nrm_msrs_len+1:nrm_msrs_len+length(msrmnts)) =...
          msrmnts;
        nrm_msrs_len = nrm_msrs_len + length(msrmnts);
        b_indx = dec_info.vid_region.blk_indx(1,:);
        fprintf('%s[%d %d %d] dur=%5.1f added %d len=%d size=%d\n',...
          prefix_k, b_indx(1), b_indx(2), b_indx(3),...
          toc(tstart), length(msrmnts), nrm_msrs_len, length(nrm_msrs));
      end
    end
  end
  nrm_msrs = nrm_msrs(1:nrm_msrs_len);
  
  if nargout > 1
    if isempty(nrm_msrs)
      nrm_prob = nrm_msrs;
      return
    end
    nrm_msrs = sort(nrm_msrs);
    N = length(nrm_msrs);
    
    % Compute inverse standard normal distribution at the points
    % (n-0.5)/N, n=1,...,N using the fact that
    %   inv_std_gauss(x) = sqrt(2)*inv_erf(2*x-1)
    nrm_prob = ((1:N)'-0.5)/N;
    nrm_prob  = sqrt(2) * erfinv(2*nrm_prob -1);
  end
  
  function initVBlocker()
    if ~isfield(dec_info,'blocker') && isfield(dec_info, 'raw_vid')...
        && ~isempty(dec_info.enc_opts)
      
      if dec_info.enc_opts.n_frames == -1
        dec_info.enc_opts.setParams(struct('n_frames',...
          dec_info.raw_vid.n_frames - dec_info.enc_opts.start_frame + 1));
      end
      
      %calculate the dimensions of the read in video
      Ysz=[...
        dec_info.raw_vid.height,...
        dec_info.raw_vid.width,...
        dec_info.enc_opts.n_frames];
      
      if dec_info.raw_vid.UVpresent
        UVsz = [dec_info.raw_vid.uvHeight(),...
          dec_info.raw_vid.uvWidth(),...
          dec_info.enc_opts.n_frames];
        
        dec_info.raw_size = [Ysz; UVsz; UVsz];
      else
        dec_info.UVblk_size= [0,0,0];
        dec_info.raw_size = Ysz;
      end
      
      dec_info.vid_blocker = VidBlocker([], dec_info.raw_vid, ...
        dec_info.enc_opts);
    end
  end
  
end
