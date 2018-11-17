function CM3 = CMTriplet(CM)

p = size(CM{1},1); % number of partition

q = 2;

CM3 = cell(1,6);

for i = 1 : 6
    CM3{i} = zeros(p-q,p-q);
end

for i = 1 : p-q
    
    for j = 1 : p-q
                        
        CM3{1}(i,j) = sum(sum(CM{1}(i:i+2,j:j+2)))/9; % 0-direction
        
        CM3{2}(i,j) = (sum(CM{1}(i,j:j+2))+sum(CM{2}(i+1,j:j+2))-sum(CM{1}(i+2,j:j+2)))/9; % x-direction
        
        CM3{3}(i,j) = (sum(CM{1}(i:i+2,j))+sum(CM{3}(i:i+2,j+1))-sum(CM{1}(i:i+2,j+2)))/9; % y-direction
                
        CM3{4}(i,j) = (sum(CM{1}(i,j:j+2))-2*sum(CM{1}(i+1,j:j+2))+sum(CM{1}(i+2,j:j+2)))/12; % xx-direction
        
        CM3{5}(i,j) = (CM{1}(i,j)+CM{1}(i+2,j+2)-CM{1}(i,j+2)-CM{1}(i+2,j)+CM{2}(i+1,j)-CM{2}(i+1,j+2)+CM{3}(i,j+1)-CM{3}(i+2,j+1)+CM{5}(i+1,j+1))/9; % xy-direction
        
        CM3{6}(i,j) = (sum(CM{1}(i:i+2,j))-2*sum(CM{1}(i:i+2,j+1))+sum(CM{1}(i:i+2,j+2)))/12; % yy-direction
                
    end
    
end

