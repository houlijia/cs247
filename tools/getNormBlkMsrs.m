function [msrs, probs] = getNormBlkMsrs(inp_list, fig_hndl, proc_opts)
  %
  % Input:
  %     inp_list - a list of coded video files
  %     fig_hndl - If not empty, a figure handle where Q-Q plot is
  %                drawn.
  %     proc_opts (Optional) - a struct with any of the following
  %         fields which impact the processing mode (other fields are
  %         ignored):
  %         prefix - (0ptional) and identifying prefix to add
  %                  before all printing. Default '<Nn>] '
  %         blk_rng - If present specifies blocks range for
  %                   processing. A 2x3 array, upper row is minimum,
  %                   lower row is maximum
  %         title - title of figure
  
  if nargin < 2
    proc_opts = struct();
  end
  
  if ~isfield(proc_opts, 'prefix')
    proc_opts.prefix = '';
  end
  
  n_step = 100;
  n_blks = 0;
  if nargout > 0
    msrs = cell(1,n_step);
    if nargout > 1
      probs = msrs;
    end
  end
  
  if ~isempty(fig_hndl)
    figure(fig_hndl);
    hold on
  end
  
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
        msrmnts(clipped_indices) = code_elmnt.mean_msr;
        msrmnts(1:code_elmnt.n_no_clip) = [];
        msrmnts = ...
          (msrmnts - code_elmnt.mean_msr)/code_elmnt.stdv_msr;
        msrmnts = sort(msrmnts);
        
        if nargout > 1 || ~isempty(fig_hndl)
          N = length(msrmnts);
          
          % Compute inverse standard normal distribution
          % at the points
          % (n-0.5)/N, n=1,...,N using the fact that
          %   inv_std_gauss(x) = sqrt(2)*inv_erf(2*x-1)
          prob = ((1:N)'-0.5)/N;
          prob  = sqrt(2) * erfinv(2*prob -1);
          if ~isempty(fig_hndl)
            plot(prob, msrmnts, 'k.','MarkerSize',4);
          end
        end
        
        n_blks = n_blks+1;
        if nargout > 0
          len_msrs = length(msrs);
          if n_blks > len_msrs;
            tmp = cell(1,len_msrs+n_step);
            tmp(1:len_msrs) = msrs;
            msrs = tmp;
            
            if nargout > 1
              tmp = cell(1,len_msrs+n_step);
              tmp(1:len_msrs) = probs;
              probs = tmp;
            end
          end
          
          msrs{n_blks} = msrmnts;
          if nargout > 1
            probs{n_blks} = prob;
          end
        end
        
        b_indx = dec_info.vid_region.blk_indx(1,:);
        fprintf('%s[%d %d %d] dur=%5.1f added %d\n',...
          prefix_k, b_indx(1), b_indx(2), b_indx(3),...
          toc(tstart), length(msrmnts));
      end
    end
    
    if nargout > 0
      msrs = msrs(1:n_blks);
      if nargout > 1
        probs = probs(1:n_blks);
      end
    end
  end
  
  function initVBlocker()
    if ~isfield(dec_info,'blocker') && isfield(dec_info, 'raw_vid')...
        && ~isempty(dec_info.enc_opts)
      
      if dec_info.enc_opts.n_frames == -1
        dec_info.enc_opts.setParams(struct('n_frames',...
          dec_info.raw_vid.n_frames - dec_info.enc_opts.start_frame + 1));
      end
      
      dec_info.vid_blocker = VidBlocker(...
        [], dec_info.raw_vid, dec_info.enc_opts);
      
    end
  end
  
end

