classdef SensingMatrix < CodeElement
    %SensingMatrix - an abstract class describing a sensing matrix
     
    properties (Constant)
        % If check_mult_eps is not empty it should be a real non-negaive.
        % In that case the actual matrix is generated and each multVec()
        % and multVecTrnsp() checks result against actual matrix
        % multiplication. differences of magnitude exceeding check_mult_eps
        % are reported.
        % Note: This can be a huge memory guzzler!
        check_mult_eps = [];
%        check_mult_eps = 1e-10;
    end
    
    properties
        n_rows=0;    % Number of rows
        n_cols=0;    % Number of columns
        
        % Zeroed rows can be set so that multLR treats the matrix as if the
        % specified rows have been zeroed.  This does not apply to multVec()
        % and multTrnspVec.
        zeroed_rows=[];
        
        % If true treat the matrix as the transpose of the original matrix
        is_transposed = false;
    end
    
    properties (Access=protected)
        matrix = [];
        msrs_normalizer = [];
        msrs_normalizer_trnsp = [];
    end
    
    methods (Access=protected)
        % Constructor.  Can be started with no arguments or with both of
        % them.
        % In all subclasses, calling the constructor with no arguments and
        % then calling with some arguments should be equivalent to
        % calling the constructor with the same arguments
        function setSensingMatrix(obj, num_rows, num_columns)
            if nargin == 3
                obj.n_rows = num_rows;
                obj.n_cols = num_columns;
            end
        end
            
    end
    
    methods
        function obj=SensingMatrix(varargin)
            obj.setSensingMatrix(varargin{:});
        end
        
        function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'num_rows','num_columns'});
            obj.setSensingMatrix(varargin{:});
        end
        
        function transpose(obj)
            obj.is_transposed = ~obj.is_transposed;
        end
        
        function ncl = nCols(obj)
            if obj.is_transposed
                ncl = obj.n_rows;
            else
                ncl = obj.n_cols;
            end
        end
        
        function nrw = nRows(obj)
            if obj.is_transposed
                nrw = obj.n_cols;
            else
                nrw = obj.n_rows;
            end
        end
        
        % encode - a method which is abstract in CodeElement
        function len=encode(obj, code_dst, ~)
            len = code_dst.writeUInt([obj.n_rows, obj.n_cols, obj.is_transposed]);
        end
        
        % decode - a method which is abstract in CodeElement
        function len=decode(obj, code_src, ~, cnt)
            if nargin < 4
                cnt = inf;
            end
            
            [vals, len] = code_src.readUInt(cnt, [3,1]);
            if ischar(vals) || (isscalar(vals) && vals==-1)
                len = vals;
                return;
            end
            obj.setSensingMatrix(vals(1), vals(2));
            if vals(3)
                obj.transpose();
            end
        end
        
        % Set the zeroed rows for multLR
        function setZeroedRows(obj, zr)
            obj.zeroed_rows = zr;
        end
        
        % multVec - Multiply a vector x of length n_cols by the matrix and return
        % a vector y of length n_rows.
        function y = multVec(obj, x)
            if obj.is_transposed;
                y = obj.doMultTrnspVec(x);
            else
                y = obj.doMultVec(x);
            end
            
            if ~isempty(obj.check_mult_eps)
                mtrx = obj.getMatrix();
                if size(mtrx,1)==0 && isempty(y)
                    err = 0;
                else
                    err = norm(y - mtrx * x, inf);
                end
                
                if err > obj.check_mult_eps
                    warning('SensingMatrix:MultErr','Error of %.3g in multiplying by %s',...
                        err, class(obj));
                end
            end
            y(obj.zeroed_rows) = 0;
        end
        
        % multTrnspVec - Multiply a vector x of length n_rows by the transpose
        % of the matrix and return A vector y of length n_cols.  The result
        % may be scaled (see below).
        function y = multTrnspVec(obj, x)
            x(obj.zeroed_rows) = 0;
            if obj.is_transposed
                y = obj.doMultVec(x);
            else
                y = obj.doMultTrnspVec(x);
            end
            if ~isempty(obj.check_mult_eps)
                mtrx = obj.getMatrix();
                err = norm(y - obj.trnspScale()*mtrx' * x, inf);
                
                if err > obj.check_mult_eps
                    warning('SensingMatrix:MultErr',...
                        'Error of %.3g in transpose multiplying by %s',...
                        err, class(obj));
                end
            end
            
        end
        
        % multLR chooses invokes multVec if mode is false and multTrnspVec  if mode
        % is true.
        function y = multLR(obj, x, mode)
            if obj.is_transposed
                mode = ~mode;
            end
            if mode
                y = obj.multTrnspVec(x);
            else
                y = obj.multVec(x);
            end
        end
        
        % Return a function handle for multLR
        function hndl = getHandle_multLR(obj)
            hndl = @(x,mode) multLR(obj, x, mode);
        end
        
        function y = multMat(obj, x)
            if obj.is_transposed
              y = obj.doMultTrnspMat(x);
            else
              y = obj.doMultMat(x);
            end
        end
        
        function y = multTrnspMat(obj, x)
            if obj.is_transposed
              y = obj.doMultMat(x);
            else
              y = obj.doMultTrnspMat(x);
            end
        end
        
        function y = doMultMat(obj, x)
            y = zeros(obj.nRows(), size(x,2));
            for k=1:size(x,2)
                y(:,k) = obj.doMultVec(x(:,k));
            end
        end
        
        function y = doMultTrnspMat(obj, x)
            y = zeros(obj.nCols(), size(x,2));
            for k=1:size(x,2)
                y(:,k) = obj.doMultTrnspVec(x(:,k));
            end
        end
        
        function mtrx = getMatrix(obj)
          if isempty(obj.matrix)
            obj.matrix = obj.compMatrix();
          end
          mtrx = obj.matrix;
          if obj.is_transposed
            mtrx = mtrx';
          end
        end
        
        function mtx = compMatrix(obj)
          mtx = zeros(obj.n_rows, obj.n_cols);
          for k=1:size(mtx,2)
            x = zeros(obj.n_cols,1);
            x(k) = 1;
            mtx(:,k) = obj.doMultVec(x);
          end
        end
        
        % returns true if getMatrix returns a sparse matrix
        function is_sprs = isSparse(~)
            is_sprs = false;
        end
        
        function setMatrix(obj, mtrx)
            obj.matrix = mtrx;
        end
        
        % Some sensing matrices have some special rows: The output from
        % these rows may have a special significance and should not be
        % clipped in the quantizer, e.g. if that output is the DC value
        % The following functions support this capability
        
        % Return the number of no-clip elements in the output vector.
        function n_no_clip=nNoClip(~)
            n_no_clip = 0;
        end
        
        % Sort the measurements vector y so that the no clip elements are first.
        % If the function is called without argument it returns a handle to
        % itself.
        function out = sortNoClip(obj, y)
            if nargin < 2
                out = @(yy) obj.sortNoClip(yy);
            else
                out = y; % Generic behavior - no sort is done
            end
        end
        
        % Unsort the sorted vector y so that the no clip elements are in
        % their original place.
        % If the function is called without argument it returns a handle to
        % itself.
        function out = unsortNoClip(obj, y)
            if nargin < 2
                out = @(yy) obj.unsortNoClip(yy);
            else
                out = y; % Generic behavior - no sort is done
            end
        end
        
        % Get an array of measurements (or modified measurements)
        % which correspond to specific  offsets.
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
        %                            ofst_msrs (see below)
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
        function [ofst_msrs] = getOffsetMsrmnts(~, ofsts,~, ~,params)
          lofst = length(ofsts);
          ofst_msrs = zeros(0,lofst);
          
          if nargin >= 5 && ...
              isfield(params,'ofsts_list') && isfield(params,'nghbr_list')
            ofst_msrs  = obj.getEdgeMsrmnts(ofst_msrs, params.ofsts_list,...
              params.nghbr_list);
          end
            
        end
        
        % Get the DC measurement
        function dc_val = getDC(~,~)
            dc_val = 0;  %default when DC is not known
        end
        
        % normalize a measurements vector, so that if the input vector
        % components are independet, identically distributed random
        % variables, the each element of y will have the same variance as
        % the input vector elements (if the matrix is transposed, the
        % operation should be changed accordingly).
        function y=normalizeMsrs(obj,y)
          if obj.is_transposed
            if isempty(obj.msrs_normalizer_trnsp)
              obj.msrs_normalizer_trnsp = obj.compMsrsNormalizer();
            end
            y = y ./ obj.msrs_normalizer_trnsp;
          else
            if isempty(obj.msrs_normalizer)
              obj.msrs_normalizer = obj.compMsrsNormalizer();
            end
            y = y ./ obj.msrs_normalizer;
          end
        end
        
        % undo the operation of normalizeMsrs
        function y = deNormalizeMsrs(obj,y)
          if obj.is_transposed
            if isempty(obj.msrs_normalizer_trnsp)
              obj.msrs_normalizer_trnsp = obj.compMsrsNormalizer();
            end
            y = y .* obj.msrs_normalizer_trnsp;
          else
            if isempty(obj.msrs_normalizer)
              obj.msrs_normalizer = obj.compMsrsNormalizer();
            end
            y = y .* obj.msrs_normalizer;
          end
        end
        
        function y = compMsrsNormalizer(obj)
          y = zeros(obj.nCols(),1);
          for k=1:length(y)
            v = zeros(obj.nRows(),1);
            v(k) = 1;
            y(k) = norm(obj.multTrnspVec(v),2);
          end
          if obj.is_transposed
            y = y * obj.trnspScale();
          else
            y = y /obj.trnspScale();
          end
        end
        
    end
    
    methods(Abstract)
        % doMultVec - Multiply a vector x of length n_cols by the matrix and return
        % a vector y of length n_rows.
        y = doMultVec(obj, x)
        
        % doMultTrnspVec - Multiply a vector x of length n_rows by the transpose
        % of the matrix and return A vector y of length n_cols.  The result
        % may be scaled (see below).
        y = doMultTrnspVec(obj, x)
        
        % Sometimes multTrnspVec may multiply the output y by a scaling
        % factor.  This function returns the scaling factor.
        y = trnspScale(obj)
        
        % returns the norm of A'A (transpose times the matrix).
        % The norm is defined as max(obj.multTrnspVec(obj.multVec(x))) over
        % all x such that max(abs(x))=1. Therefore if B=A'A the the norm is
        % Max{Sum{abs(b(i,j); over j}; over i}
        y = normAtA(obj)
        
    end
    
    methods (Static)
        % Generate matrix of a given type and arguments
        function mtrx = construct(name,args)
            mtrx = eval(name);
            mtrx.set(args);
        end
        
        % MultOrient - multiply vector x by SensingMatrix, matrix. If orientation
        % is 1, multiply by the matrix itsself; if it is 2 multiply by the
        % transpose. return output in y.
        function y = multOrient(matrix, x, orientation)
            if orientation == 1
                y = matrix.multVec(x);
            else
                y = matrix.multTransposeVec(x);
            end
        end
    end
    
    methods (Static, Access=protected)
      function ofst_msrs = getEdgeMsrmnts(ofst_msrs, ofsts_list, nghbr_list)
        prev_msrs = ofst_msrs;
        ofst_msrs = zeros(size(prev_msrs,1), length(ofsts_list));
        for k=1:length(ofsts_list)
          ofst_msrs(:,k) = prev_msrs(:,ofsts_list(k)) - ...
            (1/size(nghbr_list,2))*sum(prev_msrs(:,nghbr_list(k,:)'),2);
        end
      end
    end
    
end
    
