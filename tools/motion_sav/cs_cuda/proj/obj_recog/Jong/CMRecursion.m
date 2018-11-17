function CM2 = CMRecursion(CM,q)

p = size(CM{1},1); % number of partition

% q = 2^(n-1);

CM2 = cell(1,6);

for i = 1 : 6
    CM2{i} = zeros(p-q,p-q);
end

CM2{1}(1:p-q,1:p-q) = (CM{1}(1:p-q,1:p-q)+CM{1}(1+q:p,1:p-q)+CM{1}(1:p-q,1+q:p)+CM{1}(1+q:p,1+q:p))/4; % 0-direction

CM2{2}(1:p-q,1:p-q) = (CM{1}(1:p-q,1:p-q)-CM{1}(1+q:p,1:p-q)+CM{1}(1:p-q,1+q:p)-CM{1}(1+q:p,1+q:p))/4; % x-direction

CM2{3}(1:p-q,1:p-q) = (CM{1}(1:p-q,1:p-q)+CM{1}(1+q:p,1:p-q)-CM{1}(1:p-q,1+q:p)-CM{1}(1+q:p,1+q:p))/4; % y-direction

CM2{4}(1:p-q,1:p-q) = (CM{2}(1:p-q,1:p-q)-CM{2}(1+q:p,1:p-q)+CM{2}(1:p-q,1+q:p)-CM{2}(1+q:p,1+q:p))/4; % xx-direction

CM2{5}(1:p-q,1:p-q) = (CM{1}(1:p-q,1:p-q)-CM{1}(1+q:p,1:p-q)-CM{1}(1:p-q,1+q:p)+CM{1}(1+q:p,1+q:p))/4; % xy-direction

CM2{6}(1:p-q,1:p-q) = (CM{3}(1:p-q,1:p-q)+CM{3}(1+q:p,1:p-q)-CM{3}(1:p-q,1+q:p)-CM{3}(1+q:p,1+q:p))/4; % yy-direction

