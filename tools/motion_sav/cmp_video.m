function [ different, component, h, w, t ] = cmp_video( V1, V2 )
% cmp_video Compares tw =two video files
%   V1 and V2 are two raw video cells.
% Output:
% different: If true the videos are differnt
% component: First component (Y=1, U=2, V=3) in which a differnece occurs.
% 0 indicates size mismatch
% h, w, t - height, width, frame of pixel where first difference was found.

different = true;
component = 0;
h = 0;
w = 0;
t = 0;

if any(any(size(V1) ~= size(V2)))
    return
end

for component=1:size(V1,2)
    cmp_size = size(V1{component});
    for t=1:cmp_size(3)
        for w=1:cmp_size(2)
            for h=1:cmp_size(1)
                if V1{component}(h,w,t) ~= V2{component}(h,w,t)
                    return
                end
            end
        end
    end
end
different = 0;

end

