function [CM,J] = LocalizedOrderedSensing(I,p)

% I : original image
% J : resized original image
% p : partition number in a side ex) p = 64, 128, 256, ...
% CM1 : compressed measurements

if nargin < 2
    
    p = 128;
    
end

CM = cell(1,6);

for i = 1 : 6
    
    CM{i} = zeros(p);
    
end

J = imresize(I,[4*p,4*p]);

for i = 1 : p
    
    for j = 1 : p
        
        bl1 = sum(sum(J(1+4*(i-1):4*i-2,1+4*(j-1):4*j-2)));
        
        bl2 = sum(sum(J(3+4*(i-1):4*i,1+4*(j-1):4*j-2)));
        
        bl3 = sum(sum(J(1+4*(i-1):4*i-2,3+4*(j-1):4*j)));
        
        bl4 = sum(sum(J(3+4*(i-1):4*i,3+4*(j-1):4*j)));
        
        CM{1}(i,j) = bl1+bl2+bl3+bl4; % 0-diretion
        
        CM{2}(i,j) = bl1-bl2+bl3-bl4; % x-direction
        
        CM{3}(i,j) = bl1+bl2-bl3-bl4; % y-direction
        
        CM{4}(i,j) = CM{1}(i,j)-2*sum(sum(J(2+4*(i-1):4*i-1,1+4*(j-1):4*j))); % xx-direction
        
        CM{5}(i,j) = bl1-bl2-bl3+bl4; % xy-direction
        
        CM{6}(i,j) = CM{1}(i,j)-2*sum(sum(J(1+4*(i-1):4*i,2+4*(j-1):4*j-1))); % yy-direction
        
    end
    
end

