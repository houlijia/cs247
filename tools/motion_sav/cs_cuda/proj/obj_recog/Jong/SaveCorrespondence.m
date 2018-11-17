function SaveCorrespondence(J,VY,FileNameO,FileNameAlert,Name)

% Saving the results

% Making a red box

if strcmp(Name,'Knife')

    J(VY(2,1)-1:VY(2,1)+1,VY(1,1)-1:VY(1,4)+1,1) = 255*ones(3,VY(1,4)-VY(1,1)+3);
    J(VY(2,2)-1:VY(2,2)+1,VY(1,2)-1:VY(1,3)+1,1) = 255*ones(3,VY(1,3)-VY(1,2)+3);
    J(VY(2,1)-1:VY(2,2)+1,VY(1,1)-1:VY(1,1)+1,1) = 255*ones(VY(2,2)-VY(2,1)+3,3);
    J(VY(2,4)-1:VY(2,3)+1,VY(1,4)-1:VY(1,4)+1,1) = 255*ones(VY(2,3)-VY(2,4)+3,3);

    J(VY(2,1)-1:VY(2,1)+1,VY(1,1)-1:VY(1,4)+1,2) = zeros(3,VY(1,4)-VY(1,1)+3);
    J(VY(2,2)-1:VY(2,2)+1,VY(1,2)-1:VY(1,3)+1,2) = zeros(3,VY(1,3)-VY(1,2)+3);
    J(VY(2,1)-1:VY(2,2)+1,VY(1,1)-1:VY(1,1)+1,2) = zeros(VY(2,2)-VY(2,1)+3,3);
    J(VY(2,4)-1:VY(2,3)+1,VY(1,4)-1:VY(1,4)+1,2) = zeros(VY(2,3)-VY(2,4)+3,3);

    J(VY(2,1)-1:VY(2,1)+1,VY(1,1)-1:VY(1,4)+1,3) = zeros(3,VY(1,4)-VY(1,1)+3);
    J(VY(2,2)-1:VY(2,2)+1,VY(1,2)-1:VY(1,3)+1,3) = zeros(3,VY(1,3)-VY(1,2)+3);
    J(VY(2,1)-1:VY(2,2)+1,VY(1,1)-1:VY(1,1)+1,3) = zeros(VY(2,2)-VY(2,1)+3,3);
    J(VY(2,4)-1:VY(2,3)+1,VY(1,4)-1:VY(1,4)+1,3) = zeros(VY(2,3)-VY(2,4)+3,3);
           
    Alert = 1;
    
    fid = fopen(FileNameAlert,'w');
    fwrite(fid,Alert,'uint8');
    fclose(fid);
    
else
    
    J(VY(2,1)-1:VY(2,1)+1,VY(1,1)-1:VY(1,4)+1,1) = zeros(3,VY(1,4)-VY(1,1)+3);
    J(VY(2,2)-1:VY(2,2)+1,VY(1,2)-1:VY(1,3)+1,1) = zeros(3,VY(1,3)-VY(1,2)+3);
    J(VY(2,1)-1:VY(2,2)+1,VY(1,1)-1:VY(1,1)+1,1) = zeros(VY(2,2)-VY(2,1)+3,3);
    J(VY(2,4)-1:VY(2,3)+1,VY(1,4)-1:VY(1,4)+1,1) = zeros(VY(2,3)-VY(2,4)+3,3);

    J(VY(2,1)-1:VY(2,1)+1,VY(1,1)-1:VY(1,4)+1,2) = 255*ones(3,VY(1,4)-VY(1,1)+3);
    J(VY(2,2)-1:VY(2,2)+1,VY(1,2)-1:VY(1,3)+1,2) = 255*ones(3,VY(1,3)-VY(1,2)+3);
    J(VY(2,1)-1:VY(2,2)+1,VY(1,1)-1:VY(1,1)+1,2) = 255*ones(VY(2,2)-VY(2,1)+3,3);
    J(VY(2,4)-1:VY(2,3)+1,VY(1,4)-1:VY(1,4)+1,2) = 255*ones(VY(2,3)-VY(2,4)+3,3);

    J(VY(2,1)-1:VY(2,1)+1,VY(1,1)-1:VY(1,4)+1,3) = zeros(3,VY(1,4)-VY(1,1)+3);
    J(VY(2,2)-1:VY(2,2)+1,VY(1,2)-1:VY(1,3)+1,3) = zeros(3,VY(1,3)-VY(1,2)+3);
    J(VY(2,1)-1:VY(2,2)+1,VY(1,1)-1:VY(1,1)+1,3) = zeros(VY(2,2)-VY(2,1)+3,3);
    J(VY(2,4)-1:VY(2,3)+1,VY(1,4)-1:VY(1,4)+1,3) = zeros(VY(2,3)-VY(2,4)+3,3);
    
end

imwrite(uint8(J),FileNameO,'JPEG')


