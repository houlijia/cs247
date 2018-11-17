clear
close all
clc
%addpath(genpath('./funs'))

filename = 'barbara.tif';
I=imread(filename);
CSr = 0.2;
[row, col, rgb] = size(I);

y = GetMeasurements(I, CSr);
