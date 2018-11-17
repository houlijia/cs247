function CorrectPairs = MatchtoDB(F1,F2,G1,G2,Threshold,NCommon)

IndexPairs = CSSURFMatch(F1,F2,Threshold); % basic matching with matlab function 'matchFeatures.m'

NPair = size(IndexPairs,1);

CorrectPairs = [];

for i = 1 : NPair
    
    LocalIndex = [];
    
    for j = 1 : NPair
        
        if G1(IndexPairs(i,1),IndexPairs(j,1)) == 1 && G2(IndexPairs(i,2),IndexPairs(j,2)) == 1
            
            LocalIndex = [LocalIndex; IndexPairs(j,1)];
            
        end
        
    end
        
    NLocalIndex = size(LocalIndex,1);
    
    n = 0;
        
    if NLocalIndex >= NCommon
           
        for j = 1 : NLocalIndex
            
            for k = j : NLocalIndex
                
                if G1(LocalIndex(j),LocalIndex(k)) == 1
                    
                    n = n + 1;
                
                end
                
            end
            
        end
    
    end
    
    if n >= NCommon-1
        
        CorrectPairs = [CorrectPairs; IndexPairs(i,:)];
        
    end
    
end


% MatchedFeatureNumber = size(CorrectPairs,1);


