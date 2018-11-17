classdef BaseSparser < handle
    %   BaseSparser - Base class for sparser classes
    %
    %   A sparsing transform is a sparsifying linear transformation T which
    %   maps the signal vector of compressive sensing into a space in which the
    %   signal is sparse (the space may be of a different dimension from the
    %   original vector).  A Sparser object contains functions to perform this
    %   transform as well as some related operations, such as:
    %   *  Multiplying by the transpose of the sparsifying matrix
    %   *  Solving the minimization problem and computing the error, the
    %      Lagrangian and other paramerters
    %
    %   The minimization problem may be defined as follows.  Let v be a
    %   reference sparse vector, e.g. one derived from a signal vector x by
    %   applying the sparsifying matrix and let s be a vector of Lagrange
    %   multipliers and let beta>0 be a penalty coefficient.  The Lagrangian 
    %   is defined by;
    %
    %      l(w)=J(w) + s'*(w-v) + (beta/2)((w-v)'(w-v))
    %
    %   Where J(w) is some target function.  The object is to calculate a
    %   sparse vector w which minimizes l(w).
    %
    %   BaseSparser is a nearly trivial Sparser class which is intended to be
    %   used as a base class for more elaborate classes.  In this sparser the
    %   sparsing transformation is the unit matrix and phi(w) is the L1 norm of
    %   w, sum(|w_i|), thus the minimization probelm is fully separable, i.e.
    %   it can be solved separately for each component of w.
    
    properties
        n_sigvec   % dimension of the signal vector
        
        % A matrix for expanding the signal prior to applying the sparser
        expndr = [];
    end
    
    methods
        % Constructor
        %   Input
        %     n_svec - signal dimension (or params)
        %     expander - optional expander matrix
        function obj = BaseSparser(varargin)
          obj.set(varargin{:});
        end
        
        function set(obj, varargin)
          varargin = parseInitArgs(varargin,{'n_svec', 'expander'});
          if ~isempty(varargin) > 0
            obj.n_sigvec = varargin{1};
            if length(varargin) > 1
                obj.setExpander(varargin{2});
            end
          end
        end
        
        function setExpander(obj, expander)
            obj.expndr = expander;
            obj.n_sigvec = expander.nCols();
        end
               
        % Returns the dimension of the sparse vector - 
        function n = dimSprsVec(obj)
            if isempty(obj.expndr)
                n = obj.n_sigvec;
            else
                n = obj.expndr.nCols();
            end
        end
        
        % Apply the sparsifying transform
        %   Input
        %     first arg - (unused) this object
        %     sgnl      - The input signal (of dimension n_sigvec)
        %  Output
        %     sprs_vec  - sparse signal (of dimension dimSprsVec())
        function sprs_vec = compSprsVec(obj, sgnl)
            if ~isempty(obj.expndr);
                sgnl = obj.expndr.multVec(sgnl);
            end
            sprs_vec = obj.do_compSprsVec(sgnl);
        end
        
        function sprs_vec = do_compSprsVec(~,sgnl)
            sprs_vec = sgnl;
        end
        
        % Apply the transpose of the sparsifying transform
        %   Input
        %     first arg - (unused) this object
        %     sprs_vec  - sparse signal (of dimension dimSprsVec())
        %  Output
        %     sgnl      - The input signal (of dimension n_sigvec)
        function sgnl = compSprsVecTrnsp(obj, sprs_vec)
            sgnl = obj.do_compSprsVecTrnsp(sprs_vec);
            if ~isempty(obj.expndr);
                sgnl = obj.expndr.multTrnspVec(sgnl);
            end
        end
 
        function sgnl = do_compSprsVecTrnsp(~, sprs_vec)
            sgnl = sprs_vec;            
        end
        
        % Compute constraints error given a signal vector
        %   Input
        %     sprs_vec  - w: sparse signal (of dimension dimSprsVec())
        %     sgnl  -     x: A reference signal (of dimension n_sigvec)
        %  Output
        %     cnstrnt_err - The difference w-Tx
        function cnstrnt_err = cnstrntError(obj, sprs_vec, sgnl)
            cnstrnt_err = obj.compCnstrntErrorFromSprsRef(sprs_vec,...
                obj.compSprsVec(sgnl));
        end
        
        % Compute Target function J()
        %   Input
        %     sprs_vec  - w: sparse signal (of dimension dimSprsVec())
        %   Output
        %     trgt: J(w) - Value of target function (L1 norm)
        function trgt = target(~, sprs_vec)
            trgt = sum(abs(sprs_vec));
        end
        
        % Compute the Lagrangian
        %   Input:
        %     obj - this object
        %     sprs_vec - sparse vector
        %     cnstrnt_err - The constraint errors vector (as computed by
        %                   obj.cnstrntError)
        %     mltplr      - Vector of Lagrange multipliers
        %     beta        - penalty coefficient
        function lgr = lgrng(obj, sprs_vec, cnstrnt_err, mltplr, beta)
            lgr = obj.trgt(sprs_vec) + dot(mltplr,cnstrnt_err) + ...
                (0.5*beta) * dot(cnstrnt_err, cnstrnt_err);
        end
    
        % Compute sparse vector w (of dimension dimSprsVec()) which minimizes J(w),
        % given a reference sparser vector, Lagrange multiplier and a penalty
        % coefficient.  Specifically it finds w which minimizes:
        %    ||w||_1 + mltplr'(v - w) + (beta/2) ||v - w||_2^2
        % where v is the reference and beta is the penalty.
        % The algorithm works on each component separately and can
        % be verified by taking the derivative of J and checking separately for
        % cases of w positive, negative or zero.  Note that the algorithm is
        % independent on the way in which the constraint error where derived, hence
        % it is independent on the sparsing transform.  It is valid as long as the
        % target function J() is L1 norm.
        % Input:
        %    ~ - this object(may be used is subclasses)
        %    ref_sprs_vec - a reference sparse vector
        %    mltplr - Lagrange multipliers
        %    beta - penalty coefficient
        function sprs_vec = optimize(~, ref_sprs_vec, mltplr, beta)
            beta_inv = 1./beta;
            v = ref_sprs_vec + mltplr*beta_inv;
            sprs_vec = max((abs(v)-beta_inv),0).* sign(v);
        end
    end
    
    
    methods (Static)
      function sprsr = construct(name, args)
        sprsr = eval(name);
        sprsr.set(args);
      end
      
      % Compute constraints error, w-v
      %     sprs_vec  - w: sparse signal (of dimension dimSprsVec())
      %     ref_vec  -  v: reference sparse signal (of dimension dimSprsVec())
      %  Output
      %     cnstrnt_err - The difference
      function cnstrnt_err = cnstrntErrorFromSprsRef(sprs_vec, ref_vec)
        cnstrnt_err = sprs_vec - ref_vec;
      end
      
    end
    
end

