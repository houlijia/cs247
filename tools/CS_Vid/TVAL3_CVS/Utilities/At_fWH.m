%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% At_fWH
%
% computes an approximation for x where Ax=b. "A" is a (fat e.g. width >> height ) sensing
% matrix that is generated from shuffling the rows and columns of a
% Walsh-Hadamard matrix and choosing only the first set of rows so that the number of rows equals to the length of "b", the compressed sensed measurements. 
% A Walsh-Hadamard matrix is always a square matrix whose dimensions are integer powers of two.
%
% The algorithm assumes that "x", the signal that was compressed sensed, is sparse.
%
% See A_fWH for details on how the compressed sensed measurements were generated.
%
%
% We recover an __approximation__ of x by doing the following steps:
%
% 1) Insert the compressed sensed measurements in their original locations in the
% vector of Walsh-Hadamard transform coefficients. The other Walsh-Hadamard coefficents
% are set to zero since "x" was assumed to be sparse.  This step undoes (approximately) the random subset selection of Walsh-Hadamard
% transform coefficients.
%
% 2) Take the inverse Walsh-Hadamard transform of the Walsh-Hadamard
% transform coefficients to recover a permuted "x".
%
% 3) Unpermute the permuted "x" to recover the original "x".
%
% [Input]
%
% [Output]
%
%
% Modified by: Yenming Lai @ Bell Laboratories,
% AMSC, University of Maryland, College Park
% 07/22/2011
%
% Written by: Chengbo Li @ Bell Laboratories
% Computational and Applied Mathematics Department, Rice University
% 07/26/2010
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function signal_that_was_compressed_sensed = ...
    At_fWH(compressed_sensed_measurements, indices_of_walsh_hadamard_cofficients_to_save, ...
    permutation_of_signal_that_was_compressed_sensed)

    %calculate the size of the input
    [number_of_rows_of_compressed_sensed_measurements, ...
        number_of_columns_of_compressed_sensed_measurements]=size(compressed_sensed_measurements);

    %warn if the input is a row vector
    if number_of_rows_of_compressed_sensed_measurements == 1
        warning('b is a row vector while computing A''b.'); 
    end

    %count the number of values in the original signal that the compressed
    %sensed measurements were taken on
    length_of_signal_that_was_compressed_sensed = ...
        length(permutation_of_signal_that_was_compressed_sensed);
        
    %calculate the number that is a integer power of two that is greater
    %than or equal to the number of length of the original sparse signal
    %since this number specifies the dimension of the square walsh hadamard
    %transform used when generating the compressed sensed measurements
    dimension_of_walsh_hadamard_matrix = 2^(ceil(log2(length_of_signal_that_was_compressed_sensed)));
    
    %initialize input to inverse walsh_hadamard_transform
    walsh_hadamard_transform_coefficients = ...
        zeros(dimension_of_walsh_hadamard_matrix,number_of_columns_of_compressed_sensed_measurements);
    
    %undo scaling done by A_fWH
    scaled_compressed_sensed_measurements= compressed_sensed_measurements / ...
        sqrt(length_of_signal_that_was_compressed_sensed);
    
    %insert the walsh_hadamard_coefficients in the correct position since
    %we chose a random subset of them to to be our compressed sensed
    %measurements
    walsh_hadamard_transform_coefficients(indices_of_walsh_hadamard_cofficients_to_save,:) = ...
        scaled_compressed_sensed_measurements;
    
    %initialize output
    signal_that_was_compressed_sensed = ...
        zeros(length_of_signal_that_was_compressed_sensed,...
        number_of_columns_of_compressed_sensed_measurements);
    
    %take inverse Walsh-Hadamard transform
    output_of_inverse_walsh_hadamard_transform = ifWHtrans(walsh_hadamard_transform_coefficients);
    
    %undo the permutation of the original input signal
    signal_that_was_compressed_sensed(permutation_of_signal_that_was_compressed_sensed,:)...
        = output_of_inverse_walsh_hadamard_transform(1:length_of_signal_that_was_compressed_sensed,:);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Original Code
%
% function X = At_fWH(B, OMEGA, P)
% 
% [m,N] = size(B);
% if m == 1
%     warning('y is a row vector while computing A''y.'); 
% end
% 
% M = length(P);
% refM = 2^(ceil(log2(M)));
% FX = zeros(refM,N);
% FX(OMEGA,:) = B/sqrt(M);
% X = zeros(M,N);
% TMP = ifWHtrans(FX);
% X(P,:) = TMP(1:M,:);