function I = BGR2I(BGR)

% I : m x n x o matrix 
% BGR : mno x 1 column vector

m = 480;

n = 512; % Modify this!

o = 3;

I = zeros(m,n,o);

for i = 1 : m
    
    for j = 1 : n
        
        for k = 1 : o
            
            I(i,j,k) = BGR(n*o*(i-1)+o*(j-1)+k);
            
        end
        
    end
    
end
