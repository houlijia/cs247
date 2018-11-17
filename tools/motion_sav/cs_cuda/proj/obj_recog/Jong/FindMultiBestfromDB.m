function FindMultiBestfromDB(FileNameI,FileNameO,FileNameAlert)

% Main function for object detection and identification from compressed
% measurements CM1

% J: original source image with size 512x512
% Threshold: a parameter for matching algorithm

% CM = File2CM(FileNameCM);
% 
% J = File2J(FileNameJ);

% CM = FileNameCM;
% 
% J = FileNameJ;



fid = fopen(FileNameI,'r');
BGRSize = 480*512*3; % Modify this!
BGR = fread(fid,BGRSize,'uint8'); % Modify this!
fclose(fid);

I = BGR2I(BGR);


[CM,J] = LocalizedOrderedSensing(I,128);


Threshold = 8;

hmax = 20*(16^2); % threshold of the Hessian determinant

Alert = 0;

fid = fopen(FileNameAlert,'w');
fwrite(fid,Alert,'uint8');
fclose(fid);


[X1,S1,Theta1,G1,Otv1] = CSSURFDetector(CM,hmax,0);



[F1,vpts1] = CSSURFDescriptor(X1,S1,Theta1,Otv1);



m = 4*size(CM{1},2);

N = size(G1,1);

NCluster = 1;

D = ones(N)-G1;

IDX = kmeans(D,NCluster);% kmeans([X1' 100*D],NCluster);

PO = cell(1,NCluster);

SC = cell(1,NCluster);

FE = cell(1,NCluster);

GR = cell(1,NCluster);

for i = 1 : NCluster
    
    indx = find(IDX'==i);
    
    PO{i} = X1(:,indx);
    
    SC{i} = S1(indx);
    
    FE{i} = F1(indx',:);
    
    GR{i} = G1(indx',indx);
    
end

imwrite(uint8(J),FileNameO,'JPEG')



S = load('2_book.mat');

FindBestfromDB(PO,FE,GR,S,J,FileNameO,FileNameAlert,Threshold,'Book',4,10)

J = imread(FileNameO);


S = load('8_nokia.mat');

FindBestfromDB(PO,FE,GR,S,J,FileNameO,FileNameAlert,Threshold,'Nokia',4,15)

J = imread(FileNameO);



S = load('10_cup.mat');

FindBestfromDB(PO,FE,GR,S,J,FileNameO,FileNameAlert,Threshold,'Cup',4,8)

J = imread(FileNameO);


S = load('15_book.mat');

FindBestfromDB(PO,FE,GR,S,J,FileNameO,FileNameAlert,Threshold,'MBook',4,7)

J = imread(FileNameO);




S = load('14_knife.mat');

FindBestfromDB(PO,FE,GR,S,J,FileNameO,FileNameAlert,8,'Knife',3,5)


