function AllCM = UpProp(CM,tag)

% Recursive algorithm for up-propagating the compressed measurements into
% the upper scales


% tag = 0 : compressive sensing
% tag = 1 : full sensing

p = size(CM{1},1);

% UptoOrder = floor(log2(p)); % floor(log2(p/(tag+1)^2));
% 
% CMDoubles = cell(1,UptoOrder);
% 
% CMDoubles{1} = CM;

if tag == 0
    
    UptoOrder = floor(log2(p)); % floor(log2(p/(tag+1)^2));

    CMDoubles = cell(1,UptoOrder);

    CMDoubles{1} = CM;
    
    for i = 2 : UptoOrder
    
        CMDoubles{i} = CMRecursion(CMDoubles{i-1},2^(i-2));
    
    end

elseif tag == 1
    
    UptoOrder = floor(log2((p+3)/(tag+1)^2));

    CMDoubles = cell(1,UptoOrder);

    CMDoubles{1} = CM;
    
    for i = 2 : UptoOrder
    
        CMDoubles{i} = CMRecursion(CMDoubles{i-1},2^i);
    
    end
    
end

% Triple scale is treated in a special way

CM3 = CMTriplet(CM);

if tag == 0
    
    CMTriplets = cell(1,UptoOrder-1);

    CMTriplets{1} = CM3;
    
    for i = 2 : UptoOrder-2
    
        CMTriplets{i} = CMRecursion(CMTriplets{i-1},3*2^(i-2));
    
    end

elseif tag == 1
    
    CMTriplets = cell(1,UptoOrder-2);
    
    CMTriplets{1} = CM3;
    
    for i = 2 : UptoOrder-2
    
        CMTriplets{i} = CMRecursion(CMTriplets{i-1},i+1);
    
    end
    
end




AllCM = cell(1,2*UptoOrder-2);

AllCM{1} = CMDoubles{1};

AllCM{2} = CMDoubles{2};

for i = 2 : UptoOrder-1
    
    AllCM{2*i-1} = CMTriplets{i-1};
    
    AllCM{2*i} = CMDoubles{i+1};
    
end

