function [V0,Vx,Vy,Vxx,Vxy,Vyy] = InterpPlane(AllCM,UptoOrder,m,tag)

% Auxiliary module of OtvIntp.m

V = cell(1,5);

ind = [1 2 3 4 6 8 12 16 24 32 48 64 96 128];

for i = 1 : 6
    
    V{i} = zeros(m,m,UptoOrder);

    for n = 1 : UptoOrder

        d = 2*ind(n);% 2^n;
        
        [X,Y] = meshgrid(d:4:m-d,d:4:m-d);
        
        [Xq,Yq] = meshgrid(d:m-d,d:m-d);
        
        switch tag
            
            case 0
                
                V{i}(d:m-d,d:m-d,n) = interp2(X,Y,AllCM{n}{i},Xq,Yq,'cubic');
        
            case 1
                
                V{i}(d:m-d,d:m-d,n) = AllCM{n}{i};
                
        end
            
    end
    
end

V0 = V{1};
Vx = V{2};
Vy = V{3};
Vxx = V{4};
Vxy = V{5};
Vyy = V{6};

