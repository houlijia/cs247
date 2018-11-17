classdef SensingMatrixSqrRnd < SensingMatrixSqr & SensingMatrixRnd
    %SensingMatrixSqrRnd defines a SensingMatrixSqr where the permutations
    %are defined randomly.
    
    properties (Constant)
        % These fields should normally be false and set only for debugging.
        % If true, they override the randomization and cause generation of
        % unit permutations.
        unit_permut_L = false;
        unit_permut_R = false;
    end
    
    methods
        %Constructor
        function obj = SensingMatrixSqrRnd(num_rows, num_columns, ...
                rnd_seed, order, prmt_info, rnd_type)
            if nargin < 2
                smr_args = {};
            elseif nargin < 3
                smr_args = {num_rows, num_columns};
            elseif nargin < 6
                smr_args = {num_rows, num_columns, rnd_seed};
            else
                smr_args = {num_rows, num_columns, rnd_seed, rnd_type};
            end
            obj = obj@SensingMatrixRnd(smr_args{:});
            
            if nargin >=2
                if nargin < 5
                    prmt_info = struct();
                    if nargin < 4
                        order = obj.defaultOrder(num_rows, num_columns, prmt_info);
                    end
                end
                obj.setOrder(order,prmt_info);
            end
        end
        
        function set(obj, varargin)
            obj.setSensingMatrixSqrRnd(varargin{:});
        end
        
        function setOrder(obj, order, prmt_info)
            [PL, PR] = obj.makePermutations(order, prmt_info);
            obj.setPermutations(order, PL', PR');
        end
        
        function len=encode(obj, code_dst, info)
            % Encode basic info and randomization info
            len = obj.encode@SensingMatrixRnd(code_dst, info);
            if ischar(len)
                return;
            end
            total = len;
            
            % Encode order of square matrix
            len = code_dst.writeUInt(obj.sqr_order);
            if ischar(len)
                return;
            end
            total = len+total;
            
            len = obj.encodePermutations(code_dst, info);
            if ischar(len)
                return;
            end
            len = total+len;
        end
        
        function len=decode(obj, code_src, info, cnt)
            if nargin < 4
                cnt = inf;
            end
            
            %Decode basic info and andomization Info
            n_read = obj.decode@SensingMatrixRnd(code_src, info, cnt);
            if ischar(n_read) || n_read == -1
                if n_read == -1
                    len = 'EOD found';
                else
                    len = n_read;
                end
                return;
            end
            len = n_read;
            
            % Decode order of square matrix
            [order, n_read] = code_src.readUInt(cnt-len);
            if ischar(order) || order == -1
                if order == -1
                    len = 'EOD found';
                else
                    len = order;
                end
                return;
            end
            len = len + n_read;
            order = double(order);
            
            % Read permutations info
            [prmt_info, n_read] = obj.decodePermutations(code_src, info, cnt-len);
            if ischar(prmt_info)
                len = prmt_info;
                return;
            end
            len = len + n_read;
            obj.setOrder(order, prmt_info);
        end
    end
    
    methods(Access=protected)
        function setSensingMatrixSqrRnd(obj,num_rows, num_columns, ...
                rnd_seed, order, prmt_info, rnd_type)
            switch nargin
                case 1
                    smr_args = {};
                case 3
                    smr_args = {num_rows, num_columns};
                case {4,5,6}
                    smr_args = {num_rows, num_columns, rnd_seed};
                case 7
                    smr_args = {num_rows, num_columns, rnd_seed, rnd_type};
            end
            obj.setSensingMatrixRnd(smr_args{:});
            
            if nargin >=3
                if nargin < 6
                    prmt_info = struct();
                    if nargin < 5
                        order = obj.defaultOrder(num_rows, num_columns, prmt_info);
                    end
                end
                obj.setOrder(order,prmt_info);
            end
         end
            
        function [PL, PR] = makePermutations(obj, order, ~)
            if obj.unit_permut_L
                PL = 1:order;
            else
                PL = obj.rnd_strm.randperm(order);
            end
            if obj.unit_permut_R
                PR = 1:order;
            else
                PR = obj.rnd_strm.randperm(order);
            end
        end
        
        function len=encodePermutations(~,~,~)
            len = 0;
        end
        
        function [prmt_info, len] = decodePermutations(~, ~, ~, ~,~)
            len = 0;
            prmt_info = struct();
        end
    end
    
end

