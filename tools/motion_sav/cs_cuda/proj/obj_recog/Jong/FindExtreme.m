function [X,S,G] = FindExtreme(DoH,hmax)

r = size(DoH,1);
c = size(DoH,2);
o = size(DoH,3);

X = [];

S = [];

n = 0;

Status = ones(r,c,o);

ind = [1 2 3 4 6 8 12 16 24 32 48 64 96 128];


% D = abs(DoH(2:r-1,2:c-1,2:o-1)) > hmax;
% E = D;

% CDoH = DoH(2:r-1,2:c-1,2:o-1);
% 
% for i = -1 : 1
%     for j = -1 : 1
%         for k = -1 : 1
%             if i ~= 0 || j ~= 0 || k ~= 0
%                 D = D & (CDoH>DoH(2+i:r-1+i,2+j:c-1+j,2+k:o-1+k));
%                 E = E & (CDoH<DoH(2+i:r-1+i,2+j:c-1+j,2+k:o-1+k));
%             end
%         end
%     end
% end
% H = D | E;
% 
% ll = 0;
% X = zeros(2,sum(H(:)));
% S = zeros(1,size(X,2));
% for i = 2 : r-1
%     for j = 2 : c-1
%         for k = 2 : o-1
%             if H(i-1,j-1,k-1) == 1
%                 ll = ll + 1;
%                 X(:,ll) = [i; j];
%                 S(ll) = k+1;
%             end
%         end
%     end
% end



for k = 2 : o-1
    
    for i = 2*ind(k) : r-2*ind(k)
        for j = 2*ind(k) : c-2*ind(k)
                        
            if abs(DoH(i,j,k)) > hmax && Status(i,j,k) == 1
                                
                v = sort(reshape(DoH(i-1:i+1,j-1:j+1,k-1:k+1),27,1));
                                
                if (DoH(i,j,k) == v(27) && v(27)~=v(26)) || (DoH(i,j,k) == v(1) && v(1)~=v(2))
                    
                    Status(i:i+1,j:j+1,k:k+1) = zeros(2,2,2);
                    
                    n = n + 1;
                    
                    X = [X [i ; j]];
                    
                    S = [S k+1];
                    
                end
                
            end
        end
    end
end


X = X+1/2;

G = GraphObj(X,S);
