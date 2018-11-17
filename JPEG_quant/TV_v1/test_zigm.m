clear all
close all
clc
row = 16;
col= 16;
Phi_index = Hadamard_index_zigzag(row*col,row*col);
H0 = hadamard(row*col);

H1 = H0(Phi_index, :);

% figure;
% for nn=1:row*col
%     temp = reshape(H1(nn,:),[row,col]);
%     imagesc(temp);
%     title(nn); pause;
% end


T_dct = dct(eye(row));
T_2d = kron(T_dct, T_dct);
figure; imagesc((T_2d));

figure;
for nn=1:row*col
    temp(:,:,nn) = reshape(T_2d(nn,:),[row,col]);
    %imagesc(temp(:,:,nn));
    %title(nn); pause;
end

temp1 = reshape(temp,[row*col, row*col]);
I = DispDictionary(temp1);


out_ind = zigzag(reshape(1:row*col,[row,col]));
T_2d_ind = T_2d(out_ind,:);
figure;
for nn=1:row*col
    temp(:,:,nn) = reshape(T_2d_ind(nn,:),[row,col]);
    %imagesc(temp(:,:,nn));
    %title(nn); pause;
end
temp1 = reshape(temp,[row*col, row*col]);
I = DispDictionary(temp1);

% T_dct_L = dct(eye(row*col));
% figure;
% for nn=1:row*col
%     temp = reshape(T_dct_L(nn,:),[row,col]);
%     imagesc(temp);
%     title(nn); pause;
% end
