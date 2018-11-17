% Demonstrate how to calculate T' D' D T u without involving big matrices.

[T,Tt] = defTTt_NAE;
[D,Dt] = defDDt2;

MT = [ -1 0 0 0 1 0 0 0 ;0 -1 0 0 0 1 0 0;0 0 -1 0 0 0 1 0 ;0 0 0 -1 0 0 0 1;... 
    0 0 0 0 -1 0 0 0 ;0 0 0 0 0 -1 0 0;0 0 0 0 0 0 -1 0 ;0 0 0 0 0 0 0 -1];

X = rand(2,2,2);
X = rand(2,2,2);
x = X(:);

[U, V] = D(T(X));
Y = Tt(Dt(U, V));


[U2 V2] = D(reshape(MT*x,2,2,2));
Z2 = Dt(U2, V2);
y = MT'*Z2(:);

norm(Y(:) - y)