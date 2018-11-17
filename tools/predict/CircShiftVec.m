classdef CircShiftVec
    % ShiftVec implements circular shifts of a vector
    
    
    properties
        vec = [];
        shifts = [];
    end
    
    methods
        function obj = CircShiftVec(v,s)
            obj.vec = v;
            obj.shifts = mod(s, length(v));
        end
        
        function nc = nCols(obj)
            nc = length(obj.shifts);
        end
        
        function col = getCol(obj, k)
            ofst = obj.shifts(k);
            col = [obj.vec(1+ofst:end); obj.vec(1:ofst)];
        end
        
        function col = addAll(obj)
            col = zeros(size(obj.vec));
            for k=1:obj.nCols()
                col = col + obj.getCol(k);
            end
        end
    end
end

