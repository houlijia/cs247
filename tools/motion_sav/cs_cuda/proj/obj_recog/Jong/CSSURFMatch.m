function IndexPairs = CSSURFMatch(F1,F2,Threshold)

% 0 < Threshold <= 100

IndexPairs = matchFeatures(F1,F2,'Method','Exhaustive','MatchThreshold',Threshold,'MaxRatio',0.6,'Metric','SSD','Unique',1); % Matlab built-in function
