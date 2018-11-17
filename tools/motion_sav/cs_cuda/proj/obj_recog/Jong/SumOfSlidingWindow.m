function LongestVector = SumOfSlidingWindow(s,HaarX,HaarY)

% Subfunction for computing the longest vectors of interest points


NumberOfSample = size(HaarX,1)*size(HaarX,2);% 30;

Abscissa = zeros(1,NumberOfSample);

Ordinate = zeros(1,NumberOfSample);

Ang = zeros(1,NumberOfSample);


E = fspecial('gaussian',size(HaarX),2*s/5); % Gaussian width 2*sigma = 2*(s/5)

Ang = angle(HaarX+i*HaarY);

Ang = Ang(:);

Abscissa = reshape(HaarX.*E,1,NumberOfSample);

Ordinate = reshape(HaarY.*E,1,NumberOfSample);


LongestVector = zeros(2,1);

for theta = -pi+pi/2 : pi/2 : pi
    
    I = ones(NumberOfSample,1);
    
%     Mask = I;
    Mask = (Ang > (theta-pi/4)*I & Ang < (theta+pi/4)*I) | (Ang > (theta-pi/4)*I | Ang < (theta+pi/4-2*pi)*I) & (theta > 3*pi/4)*I;
%     Mask = (Ang > theta*I & Ang < (theta+pi/6)*I) | (Ang > theta*I | Ang < (theta+pi/6-2*pi)*I) & (theta > 5*pi/6)*I;
    
    LV = [Abscissa; Ordinate]*Mask;
            
    if norm(LV) > norm(LongestVector)
            
        LongestVector = LV;
    
    end
    
end

A = [1 0 -1 0; 0 1 0 -1];

[ii,jj] = max(LongestVector'*A);

LongestVector = A(:,jj);

