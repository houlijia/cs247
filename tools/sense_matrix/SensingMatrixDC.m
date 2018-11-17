classdef SensingMatrixDC < SensingMatrix
    % SensingMatrixDC - A sensing matrix which generates only one measurement - the DC.
    %   Detailed explanation goes here
    
    properties
      no_clip_flag = true;
    end
    
    methods
        function obj = SensingMatrixDC(varargin)
            obj.setSensingMatrixDC(varargin{:});
        end
        
        function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'num_columns'});
            obj.setSensingMatrixDC(varargin{:});
        end
        
        function y = doMultVec(~, x)
            y = sum(x);
        end
        
        function y = doMultTrnspVec(obj, x)
          y = obj.ones(obj.n_cols,1)*x;
        end
        
        function y = doMultMat(~, x)
          y = sum(x);
        end
        
        function y = doMultTrnspMat(obj, x)
          y = obj.ones(obj.n_cols,1)*x;
        end
        
        % Compute the Matlab matrix idential to this matrix
        function mtx = doCompMatrix(obj)
          if ~obj.is_transposed
            mtx = obj.ones(1, obj.n_cols);
          else
            mtx = obj.ones(obj.n_cols, 1);
          end
        end
            
        % normalize a measurements vector, so that if the input vector
        % components are independet, identically distributed random
        % variables, the each element of y will have the same variance as
        % the input vector elements (if the matrix is transposed, the
        % operation should be changed accordingly).
        function y=normalizeMsrs(obj,y)
          y = y /sqrt(obj.toFloat(obj.nCols()));
        end
          
        % undo the operation of normalizeMsrs
        function y = deNormalizeMsrs(obj,y)
          y = y *sqrt(obj.toFloat(obj.nCols()));
        end
          
        function setNoClipFlag(obj, val)
          obj.no_clip_flag = obj.toLogical(val);
          
          if val
            indcs = 1;
          else
            indcs = [];
          end
          obj.setIndcsNoClip(indcs, obj.isTransposed());
        end

        % Get an array of measurements which correspond to specific time
        % offsets.
       %   Input
        %     obj: This object
        %     ofsts: a vector of offsets of length lofst.
        %     msrs: The measurements vector
        %     inp_list: Can be n array which cotains measurement indices of the
        %               measurements to use in msrs, or the string 'all'
        %     params: Optional struct of parameters which may be of use to some
        %             subclasses. Possible arguments include:
        %               nrm - norm with which comparison is to be done.
        %               ofsts_list - a column vector of indices of columns of
        %                           ofst_msrs (see below)
        %               nghbr_list - an array of indices of columns of
        %                            ofst_msrs. The number of rows is the
        %                            same as length(params.ofsts_list
        %  Output
        %     ofst_msrs: An array of size [m, lofts]. The i-th column
        %        contains the measurements (or modified measurements)
        %        corresponding to the offsets ofsts(i).
        %
        %  Note: If params is present and has the fields ofsts_list
        %        and nghbr_list, then after computing ofst_msrs it is
        %        modified by a call
        %    ofst_msrs = obj.getEdgeMsrmnts(ofst_msrs, 
        %                                    params.ofsts_list,
        %                                    params.nghbr_list);
        %        or something functionally equivalent. 
        function [ofst_msrs] = getOffsetMsrmnts(obj, ofsts, msrs, inp_list, params)
          lofst = length(ofsts);
          if ~isnumeric(inp_list)
            inp_list = 1;
          end
          if ~isempty(obj.zeroed_rows)
            [~,zr,~] = intersect(inp_list, obj.zeroed_rows);
            inp_list(zr) = [];
          end
          
          ofst_msrs = msrs(inp_list)*obj.ones(1,lofst);
          
          if nargin >= 5 && ...
              isfield(params,'ofsts_list') && isfield(params,'nghbr_list')
            ofst_msrs  = obj.getEdgeMsrmnts(ofst_msrs, params.ofsts_list,...
              params.nghbr_list);
          end
        end
        
         % Get a list of measurments indices with specified offsets
        %   Input:
        %     obj: This object (ignored)
        %     ofsts: a vector of offsets of length lofst.
        %     inp_list: A list of input measurement numbers to use
        %               (optional and ignored).
        %   Output
        %     msrs: An array with rows of lengths lofst+1. Each row
        %     contains measurements indices such that if i is the first
        %     index in the row and j is the (k+1)-th index, then
        %       obj.PL(j)-obj.PL(i) = ofsts(k) mod(obj.sqr_order).  In
        %       our case, all rows will contain 1
        function msrs = getOffsetIndices(~, ofsts, ~)
            lofst = length(ofsts);
            msrs = obj.ones(1, lofst+1);
        end
        
        % Returns the sum of values of the measurement which contain DC value,
        % weighted by the ratio of the DC value to other components (in
        % terms of RMS), or 0 if there is no such measurement.
        %   Input:
        %     obj - this object
        %     msrs - the measurements vector
        function dc_val = getDC(~,msrs)
            dc_val = msrs(1);
        end
        
        % Set an exact value for norm. It can be computationally heavy
        function val = cmpExactNorm(obj)
          val = sqrt(obj.toCPUFloat(obj.n_cols));
        end
    end
    
    methods (Static)
      function mtx = constructDC(num_columns, trnsp, no_clip)
        % Create a SensingMatrixDC or similar.
        %   Input:
        %     num_columns - number of columns
        %     trnsp - (optional) logical. if true, transpose the matrix
        %     no_clip - (optional) logical. use with setNoClipFlag(). If
        %               true, n_no_clip is set to 1. Otherwise to 0;
        if num_columns == 1 && nargin>=3 && ~no_clip
          mtx = SensingMatrixUnit(num_columns);
        else
          mtx = SensingMatrixDC(num_columns);
          if nargin>=3
            mtx.setNoClipFlag(no_clip)
          end
          if nargin >= 2 && trnsp
            mtx.transpose();
          end
        end
      end
      
      function mtx = construct(varargin)
        % Create a SensingMatrixDC or similar.
        %   Input arguments:
        %     num_columns - number of columns
        %     trnsp - (optional) logical. if true, transpose the matrix
        %     no_clip - (optional) logical. use with setNoClipFlag(). If
        %               true, n_no_clip is set to 1. Otherwise to 0;
        mtx = SensingMatrixDC.constructDC(varargin{:});
      end
    end

    methods (Access=protected)
        function setSensingMatrixDC(obj, num_columns)
            if nargin < 2
                smr_args = {};
            else
                smr_args = {1, num_columns};
            end
            obj.setSensingMatrix(smr_args{:});
            obj.setOrthoRow(true);
            obj.setIndcsNoClip([], ~obj.isTransposed()); % for transpose case
            obj.setNoClipflag(obj.no_clip_flag);
        end
        
        function setCastIndex(obj)
          obj.setCastIndex@SensingMatrix();
          obj.no_clip_flag = obj.toLogical(obj.no_clip_flag);
        end
            
    end

end

