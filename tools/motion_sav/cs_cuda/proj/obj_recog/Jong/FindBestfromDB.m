function FindBestfromDB(PO,FE,GR,S,J,FileNameO,FileNameAlert,Threshold,Name,NCommon,NGraphLimit)

% Auxiliary function of 'FindMultiBestfromDB.m'

X = S.X;
F = S.F;
G = S.G;
Vx = S.Vx;

NCluster = size(PO,2);


if strcmp(Name,'Knife')
    
for i = 1 : NCluster
    
    ind = [];
    
    n = size(PO{i},2);
    
    for j = 1 : n
        
        if PO{i}(1,j) > 512*0.48;
            
            ind = [ind j];
            
        end
        
    end
    
    size(ind,2);
    
    PO{i} = PO{i}(:,ind);
    
    FE{i} = FE{i}(ind',:);
    
    GR{i} = GR{i}(ind',ind);
    
end

end


NumberofDB = size(F,2);


TotalBestCorrectPairs = [];

TotalBestMatchIndex = [];

for IndexofDB = 1 : NumberofDB
    
    BestCorrectPairs = [];
    
    BestMatchIndex = [];
    
    for i = 1 : NCluster
        
        CorrectPairs = MatchtoDB(FE{i},F{IndexofDB},GR{i},G{IndexofDB},Threshold,NCommon);
       
        if size(CorrectPairs,1) > size(BestCorrectPairs,1)
            
            BestCorrectPairs = CorrectPairs;
            
            BestMatchIndex = i;
                
        end
    end
    
    if size(BestCorrectPairs,1) > size(TotalBestCorrectPairs,1)
        
        TotalBestSubsetIndex = BestMatchIndex;
        
        TotalBestCorrectPairs = BestCorrectPairs;
        
        TotalBestMatchIndex = IndexofDB;
                
    end
    
end
    
NMatch = size(TotalBestCorrectPairs,1) % the number of matching feature poiints

% if NMatch >= 15
%     
%     CIndx = setdiff(1:size(PO{TotalBestSubsetIndex},2),TotalBestCorrectPairs(:,1)');
%     
%     PO{TotalBestSubsetIndex} = PO{TotalBestSubsetIndex}(:,CIndx);
%     
%     FE{TotalBestSubsetIndex} = FE{TotalBestSubsetIndex}(CIndx',:);
%     
%     GR{TotalBestSubsetIndex} = GR{TotalBestSubsetIndex}(CIndx',CIndx);
%     
% end


NGraph = size(TotalBestCorrectPairs,1);

if NGraph >= NGraphLimit
    
    X2 = X{TotalBestMatchIndex};
    X1 = PO{TotalBestSubsetIndex};
    
    X = X2(:,TotalBestCorrectPairs(:,2)');%+[0;m]*ones(1,NGraph);
    Y = X1(:,TotalBestCorrectPairs(:,1)');
    meanX = [0 1;1 0]*mean(X,2);
    meanY = [0 1;1 0]*mean(Y,2);
    
%     stdratio = sqrt(sum(sum((Y-meanY*ones(1,size(Y,2))).^2))/sum(sum((X-meanX*ones(1,size(X,2))).^2)));
    
    alpha = 1.1;
    
    VX = Vx{TotalBestMatchIndex};
    
    VY = round(meanY*ones(1,4)+alpha*(VX-meanX*ones(1,4)));
%     VY = round(meanY*ones(1,4)+alpha*stdratio*(VX-meanX*ones(1,4)));
    
    VY = max(2*ones(2,4),VY);
    VY = min(511*ones(2,4),VY);
    
    SaveCorrespondence(J,VY,FileNameO,FileNameAlert,Name)
%     DrawCorrespondence(TotalBestCorrectPairs,X1,J,Name) %,X2,m,A,meanY-A*meanX,Name)
    
end
