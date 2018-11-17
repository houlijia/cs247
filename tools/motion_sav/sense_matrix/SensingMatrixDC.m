classdef SensingMatrixDC < SensingMatrix
    % SensingMatrixDC - A sensing matrix which generates only one measurement - the DC.
    %   Detailed explanation goes here
    
    properties
        trnsp_scl = 1;
    end
    
    methods
        function obj = SensingMatrixDC(varargin)
            obj.setSensingMatrixDC(varargin{:});
        end
        
        function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'num_columns', 'tr_scale'});
            obj.setSensingMatrixDC(varargin{:});
        end
        
        function len=encode(obj, code_dst, ~)
            len = code_dst.writeUInt([obj.n_cols, obj.is_transposed]);
            if ischar(len)
                return;
            end
            total = len;
            
            len = code_dst.writeNumber(obj.trnsp_scl);
            if ischar(len)
                return;
            end
            len = total+len;
        end
          
        function len=decode(obj, code_src, ~, cnt)
            if nargin < 4
                cnt = inf;
            end
            
            [vals, n_read] = code_src.readUInt(cnt,[2,1]);
            if ischar(vals) || (isscalar(vals) && vals == -1)
                if ischar(vals)
                    len = vals;
                else
                    len = 'Unexpected end of data';
                end
                return;
            end
            len = n_read;
            ncl = vals(1);
            is_trnsp = vals(2);
            
            [tr_scl, n_read] = code_src.readNumber(cnt-len);
            if isempty(tr_scl) || ischar(tr_scl)
                if isempty(tr_scl)
                    len = 'Unexpected end of data';
                else
                    len = tr_scl;
                end
                return
            end
            len  = len + n_read;
            
            obj.setSensingMatrixDC(ncl, tr_scl);
            if is_trnsp
                obj.transpose();
            end
        end
        
        function y = doMultVec(~, x)
            y = sum(x);
        end
        
        function y = doMultTrnspVec(obj, x)
            y = ones(obj.n_cols,1)*(x*obj.trnspScale());
        end
        
        function y = trnspScale(obj)
            y = obj.trnsp_scl;
        end
         
        function y = normAtA(obj)
            y = obj.n_cols * obj.trnsp_scl;
        end
        
        % normalize a measurements vector, so that if the input vector
        % components are independet, identically distributed random
        % variables, the each element of y will have the same variance as
        % the input vector elements (if the matrix is transposed, the
        % operation should be changed accordingly).
        function y=normalizeMsrs(obj,y)
          if ~obj.is_transposed
            y = y /sqrt(obj.n_cols);
          else
            y = y / obj.trnspSacle();
          end
        end
          
        % undo the operation of normalizeMsrs
        function y = deNormalizeMsrs(obj,y)
          if ~obj.is_transposed
            y = y *sqrt(obj.n_cols);
          else
            y = y * obj.trnspSacle();
          end
        end
          
        function n_no_clip=nNoClip(~)
            n_no_clip = 1;
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
          
          ofst_msrs = msrs(inp_list)*ones(1,lofst);
          
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
        %       obj.IPL(j)-obj.IPL(i) = ofsts(k) mod(obj.sqr_order).  In
        %       our case, all rows will contain 1
        function msrs = getOffsetIndices(~, ofsts, ~)
            lofst = length(ofsts);
            msrs = ones(1, lofst+1);
        end
        
        % Get the DC measurement
        function dc_val = getDC(~,msrs)
            dc_val = msrs(1);
        end
    end

    methods (Access=protected)
        function setSensingMatrixDC(obj, num_columns, tr_scl)
            if nargin < 2
                smr_args = {};
            else
                smr_args = {1, num_columns};
            end
            obj.setSensingMatrix(smr_args{:});
            
            if nargin >= 3
                obj.trnsp_scl = tr_scl;
            end
        end
            
    end

end

