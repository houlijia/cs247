function [F,vpts] = CSSURFDescriptor(X,S,Theta,Otv)

% Computing SURF-like feature vectors

N = size(X,2);

F = zeros(5*3^2,N);

L = max(Otv{1});

m = size(Otv{1},1);

vpts = [];

ra = 2;

for i = 1 : N
    
    HalfSide = 2^((S(i)+1)/2);
    
    pos = round(X(:,i)-[0.5; 0.5])';
    
    features = zeros(5*3^2,1);
    
    t = (Theta(i)+1)*pi/2;  % t = 0, pi/2, pi, 3/2*pi from Theta = -1, 0, 1, 2
            
    R = round([cos(t) sin(t) 0 0 0; -sin(t) cos(t) 0 0 0; 0 0 abs(cos(t)) 0 abs(sin(t)); 0 0 0 cos(2*t) 0; 0 0 abs(sin(t)) 0 abs(cos(t))]);
    
    l = 0;
    
    for j = -1 : 1
        
        for k = -1 : 1
            
            newpos = pos + HalfSide*[j k];
            
            newpos = round(pos + (newpos-pos)*[cos(t) sin(t); -sin(t) cos(t)]);
            
            newpos = max([1 1],newpos);
            
            newpos = min([m m],newpos);
                        
            l = l + 1;
            
            s = round(S(i))-1; % to use the 1-lower-scale descriptions
            
            v = [Otv{2}(newpos(1),newpos(2),s); Otv{3}(newpos(1),newpos(2),s); Otv{4}(newpos(1),newpos(2),s)/ra; Otv{5}(newpos(1),newpos(2),s)/ra; Otv{6}(newpos(1),newpos(2),s)/ra];
            
            if S(i) >= s+1
                                
                s = s + 1;
                
                v2 =  [Otv{2}(newpos(1),newpos(2),s); Otv{3}(newpos(1),newpos(2),s); Otv{4}(newpos(1),newpos(2),s)/ra; Otv{5}(newpos(1),newpos(2),s)/ra; Otv{6}(newpos(1),newpos(2),s)/ra];
                
                a = S(i) - s;
                
                features(5*(l-1)+1:5*l) = R*((2-2^a)*v+(2^a-1)*v2);
                
            else
                
                s = max(1,s-1);
                
                v2 =  [Otv{2}(newpos(1),newpos(2),s); Otv{3}(newpos(1),newpos(2),s); Otv{4}(newpos(1),newpos(2),s)/ra; Otv{5}(newpos(1),newpos(2),s)/ra; Otv{6}(newpos(1),newpos(2),s)/ra];
                
                a = s - S(i);
                
                features(5*(l-1)+1:5*l) = R*((2^(1-a)-1)*v+(2-2^(1-a))*v2);
                
            end
            
%             features(5*(l-1)+1:5*l) = R*((1-a)*v+a*v2);
            
        end
        
    end
    
    F(:,i) = features;
    
    vpts = [vpts; struct('Scale',S(i),'SignOfLaplacian',[],'Orientation',Theta(i),'Location',X(i),'Metric',[],'Count',size(X,2))];
    
end

F = F';

