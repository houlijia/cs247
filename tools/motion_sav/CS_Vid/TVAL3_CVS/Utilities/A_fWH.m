function subset_of_wh_transform_coeff_of_permuted_input = ...
            A_fWH(input, indices_of_walsh_hadamard_cofficients_to_save, permutation_of_input)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% A_fWH takes compressed sensed measurements of the input with a "fat", row
% and column shuffled Walsh-Hadamard matrix.
%
% To achieve the above, the following steps happen:
%
% Let the input have length N. 
%
% 1) the input is permuted (this is equivalent to permuting the columns of
% the Walsh-Hadamard matrix)
%
% 2) a square Walsh-Hadamard matrix (whose dimensions are integer powers of
% two) is applied to the input.  The input is padded (if necessary) so that
% its length is also a integer power of two. 
%
% 3) a random subset of the Walsh-Hadamard coefficients of length K is chosen (this is
% equivalent to permuting the rows of the Walsh-Hadamard coefficient and then keeping only the first K rows thereby making the compressive sensing matrix "fat" (size K X N) 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% [Input]
%
% input - the input signal
%
% indices_of_walsh_hadamard_cofficients_to_save - which indices of the
% Walsh-Hadamard transform coefficent to save
%
% permutation_of_input -  a list of numbers that represents a permuation of
% the indices of the input
%
% [Output]
%
% subset_of_wh_transform_coeff_of_permuted_input - a subset of the
% Walsh-Hadamard coefficients of a permuted input
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Written by: Chengbo Li @ Bell Laboratories
% Computational and Applied Mathematics Department, Rice University
% 07/26/2010
%
% Modified by: Yenming Lai @ Bell Laboratories,
% AMSC, University of Maryland, College Park
% 06/15/2011
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %set which implementation of the Walsh Hadamard to use
    which_implementation_of_walsh_hadamard_to_use='C_version'; %'C_version', 'MATLAB'
   
    %get the dimension of in the input
    [number_of_rows_of_input, number_of_columns_of_input]=size(input);

    % warn if the input is a row
    if number_of_rows_of_input == 1
        warning('x is a row vector while computing Ax.'); 
    end

    %permute the input
    permuted_input=input(permutation_of_input,:);

    %apply walsh hadamard tranform 
    switch which_implementation_of_walsh_hadamard_to_use
        
        case 'C_version'
            walsh_hadamard_transform_of_input = fWHtrans(permuted_input);
        case 'MATLAB'
            walsh_hadamard_transform_of_input=fwht(permuted_input);
        otherwise
            error('unknown which_implementation_of_walsh_hadamard_to_use=%s\n',which_implementation_of_walsh_hadamard_to_use)
    end
    
    %scale the transform
    scaled_walsh_hadamard_transform_of_input=walsh_hadamard_transform_of_input*...
        sqrt(number_of_rows_of_input);
    
    %return only a subset of the  walsh hadamard coefficients
    subset_of_wh_transform_coeff_of_permuted_input = ...
        scaled_walsh_hadamard_transform_of_input(indices_of_walsh_hadamard_cofficients_to_save,:);

end