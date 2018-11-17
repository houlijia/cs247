function Otv = OtvIntp(AllCM,tag)

% Interpolation module from compressed measurements to complete scale space

UptoOrder = size(AllCM,2);

p = size(AllCM{1}{1},1);

switch tag
    
    case 0
        
        m = 4*p;
        
    case 1
        
        m = p+3;
        
end

[V0,Vx,Vy,Vxx,Vxy,Vyy] = InterpPlane(AllCM,UptoOrder,m,tag); % Each plane is interpolated

Otv0 = V0; % 0-direction

Otvx = Vx; % x-direction

Otvy = Vy; % y-direction

Otvxx = Vxx; % xx-direction

Otvxy = Vxy; % xy-direction

Otvyy = Vyy; % yy-direction


Otv = cell(1,6);

Otv{1} = Otv0;
Otv{2} = Otvx;
Otv{3} = Otvy;
Otv{4} = Otvxx;
Otv{5} = Otvxy;
Otv{6} = Otvyy;

