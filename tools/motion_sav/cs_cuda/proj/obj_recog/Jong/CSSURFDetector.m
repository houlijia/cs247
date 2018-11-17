function [X,S,Theta,G,Otv] = CSSURFDetector(CM,hmax,tag)

% Computing SURF-like feature locations from compressed measurements


AllCM = UpProp(CM,tag); % Getting all scale box filtering from compressed measurements


w = 0.5774; % Adusting parameter in Hessian determinant: determinant of Hessian = Hxx*Hyy-w^2*Hxy


Otv = OtvIntp(AllCM,tag); % Interpolation of box filterings



DoH = Otv{4}.*Otv{6}-w^2*Otv{5}.^2; % Determinant of Hessian (DoH)



[X,S,G] = FindExtreme(DoH,hmax); % Finding local extremal

Theta = ones(1,size(X,2));
% Theta = GetOrientation(X,S,Otv{1},Otv{2})/(pi/2); % Local orientation of feature points

