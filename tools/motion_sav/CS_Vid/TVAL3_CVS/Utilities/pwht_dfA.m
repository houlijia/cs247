function [subset_of_wh_transform_coeff_of_permuted_input]=pwht_dfA(...
    input, ...
    indices_of_walsh_hadamard_cofficeints_to_save, ...
    permuation_of_input,...
    walsh_hadamard_matrix_orientation)
% applies either the Walsh-Hadamard or the transposed
% Walsh-Hadamard transform to the given input. The input will be permuted.
% before the Walsh-Hadamard transform and only a subset of the
% transform coefficients will be returned
%
% The function name "pwht_dfA" possibly means "pick Walsh-Hadamard transform for
%   the definition of A, the restricted isometry property
%   measurement matrix"
%
% [Input]
%
% input - the input signal
%
% indices_of_walsh_hadamard_cofficeints_to_save - which indices of the
% Walsh-Hadamard transform coefficent to save
%
% permuation_of_input -  a list of numbers that represents a permuation of
% the indices of the input
%
% walsh_hadamard_matrix_orientation -  either 1 or 2
%
%   1 - use the default Walsh-Hadamard transform
%
%   2-  use the transposed Walsh-Hadamard transform
%
% [Output]
%
% subset_of_wh_transform_coeff_of_permuted_input - a subset of the
% Walsh-Hadamard coefficients of a permuted input
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if walsh_hadamard_matrix_orientation
    subset_of_wh_transform_coeff_of_permuted_input = ...
        At_fWH(input, indices_of_walsh_hadamard_cofficeints_to_save, permuation_of_input);
else
    subset_of_wh_transform_coeff_of_permuted_input = ...
        A_fWH(input, indices_of_walsh_hadamard_cofficeints_to_save, permuation_of_input);
end


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% original code by Cheng Bo Li below
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% % dfA
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function y = pwht_dfA(x,picks,perm,mode)
% switch mode
%     case 1
%         y = A_fWH(x,picks,perm);
%     case 2
%         y = At_fWH(x,picks,perm);
%     otherwise
%         error('Unknown mode passed to f_handleA!');
% end


