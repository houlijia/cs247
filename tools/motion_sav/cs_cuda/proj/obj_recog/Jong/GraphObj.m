function G = GraphObj(X,S)

m = size(X,2);

G = zeros(m);

k = 1;

S = k*S;


for i = 1 : m
    
    G(i,i:m) = double(sqrt(sum((X(:,i)*ones(1,m-i+1)-X(:,i:m)).^2)) < (2^((S(i)+1)/2)*ones(1,m-i+1)+2.^((S(i:m)+1)/2)));
    
    G(i,i) = 0;
    
end



% for i = 1 : m
%     
%     for j = i : m
%         
%         if norm(X(:,i)-X(:,j)) < 2^((S(i)+1)/2) + 2^((S(j)+1)/2)
%             
%             G(i,j) = 1;
%             
%         end
%         
%     end
%     
%     G(i,i) = 0;
%     
% end

G = (G+G');

