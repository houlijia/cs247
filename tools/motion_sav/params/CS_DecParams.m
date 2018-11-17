classdef CS_DecParams < ProcessingParams
    %CS_DecParams specifies decoder parameters

    properties (Constant=true)
        WND_HANNING = CS_EncParams.WND_HANNING;
        WND_TRIANGLE = CS_EncParams.WND_TRIANGLE;
         
        wnd_types = [CS_DecParams.WND_HANNING, CS_DecParams.WND_TRIANGLE];
    end
    
    properties
      % NOTE: properties marked as DEPRECATED are used only in the legacy
      % TVAL3_CVS_D2_a.m function
      
      % Window type to use when combining blocks with overlap
      wnd_type = CS_DecParams.WND_HANNING;
      
      % Expansion level
      expand_cnstrnt_level = 0;
      
      % max # of outer iterations
      max_out_iters = 128;
      
      % max # of inner iterations
      max_int_iters = 4;
      
      % max total number of iterations
      max_iters = 512;
      
      % If step size magnitude decreases from one internal iteration to the
      % next by a factor of less than step_ratio_thrsh, stop inner
      % loop.
      step_ratio_thrsh = 0.75;
      
      % Epsilon thresholds
      eps_dLd = 1e-4;
      
      % Iteration end thresholds:
      % - Thresholds for Lagrangian change
      eps_lgrng_chg_init = 0.1; % Initial threshold
      eps_lgrng_chg_rate = 0.5; % Scaler of threshold when reached.
      eps_lgrng_chg = 0.05;   % Lowest threshold
      
      % - Thresholds for constraints errors
      eps_A_maxerr = 3;
      eps_A_sqrerr = 2;
      eps_D_maxerr = 0.2;
      eps_D_sqrerr = 0.01;
      
      % if not zero, display results when external interation number is 
      % is divisble by disp
      disp = 0;
      
      % if not empty, and reference for computing PSNR is available, 
      % display results when a dB value in the list is reached.
      disp_db = [];
      
      % Method of initialization of x:
      % 0: x=0
      % 1: x=A'b,
      % -1: Before calling the solve routine, replace init by a
      %     reference block and set x to the reference block
      init = 0;
      
      % For solveQuant only. 
      
      % - Start with an approximate solution produced by
      %   solveExact, with all threshold relaxed by a factor of
      %   solve_exact_fctr. 0 means do not initialize solve_exact.
      solve_exact_fctr=0;
      
      % - Transition control parameters. The transition width is the sum of
      %   a component determined by the allowed error (maxAerr) and a
      %   component determined by the standard deviation of the
      %   measurements noise.
      q_trans_Aerr = 0.01;  % scaler to allowed error
      q_trans_msrs_err = 1; % scaler to measurements error;
      
      % Reference video for comparison
      ref = [];
      
      use_old = 0;
      
      % If Q_msrmnts is true the decoder is solveQuant()
      Q_msrmnts = 0;
      
      % Determine which parsifier to use and with what parameters.
      sparsifier = struct('type', 'VidSparserTV_DCT', ...
        'args', struct('mode',0));
      
      % beta parameters:
      % - rate control for penalty coefficients. Recommended > 2
      beta_rate = 2;
      beta_rate_thresh = 0.2;  % Used only by TVAL3_CVS_D2_a.m
      
      % -start and final
      beta_A0 = 0.1;
      beta_D0 = 0.1;
      beta_A=inf;
      beta_D=inf;
      
      
      % Line search options: Setting Armijio condition constant.
      lsrch_c = 1.e-5;
      lsrch_alpha_rate = 0.6;
      
      % Control the degree of nonmonotonicity. 0 corresponds to monotone line search.
      % The best convergence is obtained by using values closer to 1 when the iterates
      % are far from the optimum, and using values closer to 0 when near an optimum.
      lsrch_wgt = 0.99;
      lsrch_wgt_rate = 0.9;
      
      % Levels of comparison of decoded frames to original frames:
      %            0: No frame by frame comparison
      %            1: Print frame by frame PSNR
      %            2: Print frame by frame PSNR and display frames next
      %               to each other.
      cmpr_frm_by_frm = 1;
      
      % If true compare PSNR block by block. This comparison is on the
      % reconstructed pixel vector (before expansion).
      cmpr_blk_by_blk = 1;
      
      % If true compute reconstruction statistics
      cmp_solve_stats = false;
    end
    
    methods
      function obj=CS_DecParams(def)
        if nargin > 0
          args = {def};
        else
          args = {};
        end
        obj = obj@ProcessingParams(args{:});
        
        if ~any(obj.wnd_types == obj.wnd_type)
          error('Illegal wnd_type: %d', obj.wnd_type);
        end
      end
      
      function str = classDescription(~)
        str = 'Decoding options';
      end
      
     end
    
end

