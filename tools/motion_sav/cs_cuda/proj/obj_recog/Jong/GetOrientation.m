function Theta = GetOrientation(X,S,OtvHaarX,OtvHaarY)

% Angle of the longest vector of interest points

[r,c] = size(OtvHaarX(:,:,1));

HalfSide = 2.^((S+1)/2);

N = size(X,2);

LongestVector = zeros(2,N);

for n = 1 : N
    
    x = X(:,n);
    sigma = HalfSide(:,n);
    s = round(S(n));% S(n);
    
    intvx = max(1,round(x(1)-sigma)):min(r,round(x(1)+sigma)); % width 4*sigma = 4*(s/5) = s*(4/5)
    intvy = max(1,round(x(2)-sigma)):min(c,round(x(2)+sigma)); % width 4*sigma = 4*(s/5) = s*(4/5)
    LongestVector(:,n) = SumOfSlidingWindow(sigma,OtvHaarX(intvx,intvy,s-1),OtvHaarY(intvx,intvy,s-1));
    
end

Theta = angle(LongestVector(1,:)+i*LongestVector(2,:));

% Theta = ones(1,N);