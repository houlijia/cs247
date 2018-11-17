classdef SensingMatrixDFT  < SensingMatrixSqrRnd
    % SensingMatrixDFT Implementes a sensing matrix based on random selection of
    % rows and columns from a DFT transform of appropriate order
   
    properties
        % Determines whether to use Matlab' built in fast WH functions or
        % the .mex files.
        log2order;
    end
    
    
    methods
        % Constructor
        %   Input:
        %     num_rows - number of rows, or a struct that has a field
        %         'N_msrs', which specifies the number of rows.
        %     num_columns - number of columns
        %     rnd_seed - random number generation seed
        %     order - order of the Walsh Hadamard matrix (power of 2).
        %            if not specificed the lowest possible value is used.
        %     rnd_type - type of random number generator. Can be
        %        string - RandStream type
        %        0 - default type
        %        1 - Use RndCStrm
        %        2 - Use RndCStrm and compare with RandStream
         function obj = SensingMatrixDFT(varargin)
            obj.setSensingMatrixDFT(varargin{:})
        end
        
        % Set Initialize
        %   Input:
        %     obj - this object
        %     num_rows - number of rows, or a struct that has a field
        %         'N_msrs', which specifies the number of rows.
        %     num_columns - number of columns
        %     rnd_seed - random number generation seed
        %     order - order of the Walsh Hadamard matrix (power of 2).
        %            if not specificed the lowest possible value is used.
        %     rnd_type - type of random number generator. Can be
        %        string - RandStream type
        %        0 - default type
        %        1 - Use RndCStrm
        %        2 - Use RndCStrm and compare with RandStream
        function set(obj, varargin)
            varargin = parseInitArgs(varargin, {'num_rows', 'num_columns', ...
                'rnd_seed', 'order', 'rnd_type'});
            obj.setSensingMatrixDFT(varargin{:});
        end
        
        function n_no_clip=nNoClip(~)
            n_no_clip = 1;
        end

        % Override the same function of the superclass in order to insure that
        % the first measurement is selected.  Make sure that the first
        % entry of the output is the DC, i.e. IPL(1)=1
        function setOrder(obj, order, ~)
            [PL, PR] = obj.makePermutations(order);
            obj.log2order = nextpow2(order);
            obj.setPermutations(order, PL, PR');
        end
                
        function y=multSqr(obj,x)
            if ~isa(x, 'double')
                x = double(x);
            end
            
            if obj.sqr_order == 1
                y = x;
                return
            end
            s2 = obj.sqr_order/2;
            z = fft(x(:),obj.sqr_order);
            y = [real(z(2:s2)), imag(z(2:s2))]';
            y = [real(z(1)); real(z(s2+1)); y(:)];
            
%             % test that multTranspSqr() returns the original vector
%             x_tst = obj.multTrnspSqr(y);
%             err = norm(x-x_tst,1);
%             if err > 1e-8 * norm(x,1)
%               error('reverse does not work');
%             end
        end
        
        function y = multTrnspSqr(obj,x)
            if obj.sqr_order == 1
                y = x;
                return
            end
            z = complex(x(3:2:obj.sqr_order), x(4:2:obj.sqr_order));
            zc = conj(z);
            zc = zc(end:-1:1);
            z = [complex(x(1),0); z(:); complex(x(2),0); zc(:)];
            y= real(ifft(z));
        end
            
        function y = trnspScale(obj)
            y = 1./double(obj.sqr_order);
        end

        function y = normAtA(~)
            y = 1;
        end
                
        function order = defaultOrder(~, num_rows, num_columns, ~)
            order = pow2(nextpow2(max(num_columns, num_rows)));
        end
        
        % Get the DC measurement
        function dc_val = getDC(~,msrs)
            dc_val = msrs(1);
        end
         
    end
    
    methods(Access=protected)
        % Set Initialize
        %   Input:
        %     obj - this object
        %     num_rows - number of rows, or a struct that has a field
        %         'N_msrs', which specifies the number of rows.
        %     num_columns - number of columns
        %     rnd_seed - random number generation seed
        %     order - order of the Walsh Hadamard matrix (power of 2).
        %            if not specificed the lowest possible value is used.
        %     rnd_type - type of random number generator. Can be
        %        string - RandStream type
        %        0 - default type
        %        1 - Use RndCStrm
        %        2 - Use RndCStrm and compare with RandStream
         function setSensingMatrixDFT(obj,num_rows, num_columns, ...
                rnd_seed, order, rnd_type)
            if nargin < 3
                smr_args = {};
            else
                if isstruct(num_rows)
                    num_rows = num_rows.N_msrs;
                end
                switch nargin
                    case 3
                        smr_args = { num_rows, num_columns };
                    case 4
                        smr_args = { num_rows, num_columns, rnd_seed};
                    case 5
                        smr_args = { num_rows, num_columns, rnd_seed, order};
                    case 6
                        % Inserting struct() for prmt_info
                        smr_args = { num_rows, num_columns, rnd_seed, ...
                            order, struct(), rnd_type};
                end
            end
            obj.setSensingMatrixSqrRnd(smr_args{:});
        end 
        
        function [PL, PR] = makePermutations(obj, order, ~)
            PL = obj.makeRowSlctPermutation(order);
            PR = obj.rnd_strm.randperm(order);
        end
        
        function PL = makeRowSlctPermutation(obj,order, nrw)
            if nargin < 3
                nrw = obj.n_rows;
            end
            if order > 2
                m = order/2 - 1;
                p = obj.rnd_strm.randperm(m);
                p = [2*p; 2*p+1];
                if mod(nrw,2)
                    PL = [1; order; p(:)];
                else
                    PL = [1; 2; p(:)+1];
                end
            else
                PL = (1:order)';
            end
        end            

        % Organize measerements as complex numbers. The measuerement
        % operation, y=obj.multVec(x) includes three steps:
        % * Zero padding x to the DFT order N and computing the DFT vector X.
        % * Ordering the real and imaginary parts as follows (with zero
        %   based indexing:
        %   [Re(X(0)),Re(X(N/2)),Re(X(1),Im(X(1)),...,Re(X(N/2-1),Im(X(N/2-1))]'
        % * Selecting a subset of the created vector.
        %   Input
        %     obj: This measurements matrix
        %     msrs: A vector of measurements (subset of the measurements
        %           produced by multiplying an input vector by obj).
        %     inp_list: An array which contains the indices of measurements
        %               in msrs. If y=obj.multVec(x)  and msrs is a subset
        %               of y, then
        %                  msrs == y(inp_list).
        %               If inp_list is omitted it is assumed to be  
        %               1:obj.nRows(), i.e. msrs == y.
        %  Output
        %    tlist: The entries of y are real or imignary parts of X. 
        %           tlist is a list of indices of entries in X whose 
        %           real and imaginary parts are entries in msrs 
        %           (or the imaginary part is always zero).
        %           The entries in tlist are indices starting from zero.
        %           They are organized in such a way that the indices of
        %           coefficients which are always real (0,N/2), if present,
        %           are first.
        %    cmsrs: The vector v(tlist)
        %    nrl:   Number of coefficients which are always real.
        function [tlist, cmsrs, nrl] = sortMsrsList(obj, msrs, inp_list)
            tlist = zeros(2,1);
            cmsrs = complex([0;0],[0;0]);
            nrl = 0;
            
            ipl_list = obj.IPL(inp_list);
            
            % Special case of X(0) and X(N/2), which are always real
            rl_only_ordr = [0,obj.sqr_order/2];
            for k=1:2
                i1 = find(ipl_list == k, 1);
                if ~isempty(i1)
                    nrl = nrl+1;
                    tlist(nrl) = rl_only_ordr(k);
                    cmsrs(nrl) = complex(msrs(inp_list(i1)),0);
                end
            end
            
            % Meaning of even and odd is reversed becasue indices start
            % from 1
            rl = find(mod(ipl_list,2)==1 & ipl_list > 2);
            im = find(mod(ipl_list,2)==0 & ipl_list > 2);
            rl1 = rl+1;
            if ~isequal(rl1, im)
                % Need to sort and map
                [im, ind_rl, ~] = intersect(rl1, im);
                rl = rl(ind_rl);
            end
            
            tlist = [tlist(1:nrl); ((ipl_list(im))/2 -1)];
            % The complex part is divided by sqrt(2) to undo the
            % multiplication by sqrt(2) when measurements were created and
            % converted into real.
            cmsrs = [cmsrs(1:nrl); complex(msrs(inp_list(rl)), msrs(inp_list(im)))];
       end
    end
    
    methods (Static, Access=protected)
        function wgts = getShiftWgts(log2order)
            persistent seqs;
            
            if isempty(seqs)
                seqs = cell(1,32);
            end
            
            if isempty(seqs{log2order})
                order = 2^log2order;
                theta = (2*pi/order) * 1i;
                seqs{log2order} = exp(theta*(0:(order-1))');
            end
            wgts = seqs{log2order};
        end
        
        % Convert a comples measurements array to a real array, putting the
        % measurements which are always real (0 and N/2 indices) first, and
        % then organizing each complex row as a real row followed by a
        % complex row.
        %   Input
        %     c_msrs - a complex measurements array. Each row corresponds
        %     to one specific DFT coefficient.
        %     nrl:   Number of coefficients which are always real.
        %   Output
        %     msrs - real array
        function msrs = convertToReal(c_msrs, nrl)
           vc = c_msrs(nrl+1:end,:);
           v = [real(vc(:))'; imag(vc(:))'];
           msrs = [real(c_msrs(1:nrl,:)); ...
               reshape(v(:), 2*(size(c_msrs,1)-nrl), size(c_msrs,2))];
        end
        
        % Get the size of the matrix which would be created by
        % convertToReal
        %   Input
        %     c_size - size of the complex matrix
        %     nrl:   Number of coefficients which are always real.
        %   Output
        %     r_size - size of the real array
        function r_size = convertToRealSize(c_size, nrl)
          r_size = c_size;
          r_size(1) = 2*c_size(1) - nrl;
        end
    end
    
end

