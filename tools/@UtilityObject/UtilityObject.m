classdef UtilityObject
% contains general use methods
%
    methods (Static=true)
        
        function [list_of_smaller_value, list_of_larger_value]=...
                splitInputIntoSmallerValuesAndLargerValues...
                (input, threshold)
       % splitInputIntoSmallerValuesAndLargerValues sorts the value of the
       % input into two sets, a set of smaller values and a set of larger
       % values. The size of the sets are determined by the threshold. If a
       % value is greater than or equal to the threshold, the value is
       % placed in the larger value set, else it is placed in the smaller
       % value set
       %
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       %
       % [Input]
       %
       % input - the input, a vector
       % 
       % threshold - a real scalar, if the input value is greater than or
       % equal to this value it is placed in the larger value set,
       % otherwise it is place is in the smaller value set
       %
       % [Output]
       %
       % list_of_smaller_value - a column containing all the input values
       % smaller than a threshold
       %
       % list_of_larger_value - a column containg all the input values
       % greater than or equal to the threshold
       % 
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
           %validate inputs
           input_parser = inputParser;   
           addRequired(input_parser, 'input', ...
               @(x)validateattributes(x, {'numeric'}, {'vector'}));
           addRequired(input_parser, 'threshold', ...
               @(x)validateattributes(x, {'numeric'}, {'scalar'}));
           parse(input_parser, input, threshold); 
          
           %calculate the number of input values
           number_of_input_values=length(input);
           
           %preallocate space for list of smaller and larger values
           list_of_smaller_value_padded=777*ones(number_of_input_values,1);
           list_of_larger_value_padded=777*ones(number_of_input_values,1);

           %initialize counters
           count_of_number_of_smaller_value=0;
           count_of_number_of_larger_value=0;
          
           %iteratively sort the contents of the input
           for index_of_input=1:number_of_input_values
               ith_input_value=input(index_of_input);
               if ith_input_value >= threshold
                   count_of_number_of_larger_value = count_of_number_of_larger_value +1;
                   list_of_larger_value_padded(count_of_number_of_larger_value)=ith_input_value;
               else
                   count_of_number_of_smaller_value = count_of_number_of_smaller_value +1;
                   list_of_smaller_value_padded(count_of_number_of_smaller_value)=ith_input_value;                   
               end
           end
           
           %remove padding
           list_of_smaller_value=list_of_smaller_value_padded(1:count_of_number_of_smaller_value);
           list_of_larger_value=list_of_larger_value_padded(1:count_of_number_of_larger_value);
            
        end
        
        function [list_of_values_in_specified_range]=...
                selectValuesInSpecifiedRange(input, min_value_of_range, max_value_of_range)
        % selectValuesInSpecifiedRange extracts from the input all the
        % values which are in the specified range (closed interval) e.g. values which are >=
        % the  mininum value of range and <= the maximum value or range are
        % selected
        %
        % [Input]
        %
        % input - a numeric vector
        %
        % min_value_of_range - a real scalar, represents the mininum value
        % a selected value can take
        %
        % max_value_of_range - a real scalar, reprsents the maximum value
        % a selected value can take
        %
        % [Output]
        %
        % list_of_values_in_specified_range - a column of values of the input which lie
        % in the specified range
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
            %validate inputs
            input_parser = inputParser;   
            addRequired(input_parser, 'input', ...
            @(x)validateattributes(x, {'numeric'}, {'vector'}));
            addRequired(input_parser, 'min_value_of_range', ...
            @(x)validateattributes(x, {'numeric'}, {'scalar'}));
            addRequired(input_parser, 'max_value_of_range', ...
            @(x)validateattributes(x, {'numeric'}, {'scalar','>=',min_value_of_range}));        
            parse(input_parser, input, min_value_of_range, max_value_of_range); 
            
            number_of_values_of_input=length(input);
            list_of_values_in_specified_range_padded=777*ones(number_of_values_of_input,1);
            count_of_selected_values=0;
            
            for index_of_input_values=1:number_of_values_of_input
                ith_value=input(index_of_input_values);
                if (ith_value >= min_value_of_range) && (ith_value <= max_value_of_range)
                    count_of_selected_values=count_of_selected_values + 1;
                    list_of_values_in_specified_range_padded(count_of_selected_values)=ith_value;
                end
            end
            
            list_of_values_in_specified_range=list_of_values_in_specified_range_padded(1:count_of_selected_values);
        end
        
        function [min_value_of_range, max_value_of_range]=askUserToSpecifyARange(default_min_value, default_max_value)
            
           min_value_of_range=input(['Enter the minimum value of the range(' num2str(default_min_value) '):\n']);
           if isempty(min_value_of_range)
              fprintf('--> No user input, default min value=%d chosen\n',default_min_value);
              min_value_of_range=default_min_value; 
           end
           validateattributes(min_value_of_range,{'numeric'},{'scalar','real','nonempty'})

           max_value_of_range=input(['Enter the maximum value of the range(' num2str(default_max_value) '):\n']);
           if isempty(max_value_of_range)
             fprintf('--> No user input, default max value=%d chosen\n',default_max_value);
              max_value_of_range=default_max_value; 
           end    
           validateattributes(max_value_of_range,{'numeric'},{'scalar','real','>=',min_value_of_range})
                       
        end
        
        function [updated_values]=removeValue( list_of_value, value_to_remove)
        %removeValue removes the first instance of the value from the list of value if the value is found
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % [Input]
        %
        % list_of_value - a column vector of numbers
        %
        % value_to_remove - a number, the value to remove
        %
        % [Output]
        %
        % update_values - a copy of the input list_of_value with the
        % specified value removed
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        validateattributes(list_of_value,{'numeric'},{'column'},'removeValue','list_of_value',1)
        validateattributes(value_to_remove,{'numeric'},{'scalar'},'removeValue','value_to_remove',2)

        
        number_of_values=length(list_of_value);
        
        for index_of_number_of_values=1:number_of_values
            ith_value=list_of_value(index_of_number_of_values);
            
            if ith_value == value_to_remove
                %value found, remove it
                values_before_value_to_remove=list_of_value(1:index_of_number_of_values - 1);
                values_after_value_to_remove=list_of_value(index_of_number_of_values + 1: end);
                updated_values=[values_before_value_to_remove;values_after_value_to_remove];
                fprintf('The value to remove=%d was successfully removed.\n',value_to_remove)
                return
            end
            
        end
           
        %value not found
        fprintf('The value to remove=%d was not found and was not removed.\n',value_to_remove)
        updated_values=list_of_value;
        
        end
        
        
        function [list_of_cdf_values]=myNormCdf(input_values,mean_nrml,stdv_nrml)
        % duplicates the functionality of MATLAB's normcdf function

%             %validate inputs
%             input_parser= inputParser;
%             input_parser.addRequired('input_values', ...
%             @(x) validateattributes(x,{'numeric'},{'vector','nonempty'}));
%             input_parser.addRequired('mean_nrml', ...
%             @(x) validateattributes(x,{'numeric'},{'scalar'}));
%             input_parser.addRequired('stdv_nrml', ...
%             @(x) validateattributes(x,{'numeric'},{'scalar','positive'}));            
%             input_parser.parse(input_values,mean_nrml,stdv_nrml); 

            %calculate cdf values
            list_of_cdf_values=0.5*(1+erf((input_values-mean_nrml)./(stdv_nrml*sqrt(2))));

        end
        
        function [ell_infinity_error]=calculateLInfinityError(input_1,input_2)
        %calculates the l infinity error (max error) between the two inputs
        %
        % Input: 
        %
        % input_1 - the first input, should be numeric 
        %
        % input_2 - the second input, should be numeric
        %
        % Output:
        %
        % ell_infinity_error - the max of magnitude of the differences
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %calculate all the pointwise differences between input 1 and input 2
        difference_vector=UtilityObject.calculateDifferenceVector(input_1,input_2);

        %take the magnitude
        magnitude_of_difference_vector=abs(difference_vector);

        %calculate the ell_infinity_error
        ell_infinity_error=max(magnitude_of_difference_vector);

        end

        function [difference_vector]=calculateDifferenceVector(input_1,input_2)
            
            
            UtilityObject.checkIfInputsAreOfTheSameDimension(input_1,input_2)
            
            %convert the input into double
            input_1_as_double=double(input_1);
            input_2_as_double=double(input_2);

            %calculate the differnce
            difference_matrix=input_1_as_double-input_2_as_double;

            %flatten the matrix
            difference_vector=difference_matrix(:);        
            
        end

        function [l2_error]=calculateL2Error(input_1,input_2)

            %calculate all the pointwise differences between input 1 and input 2
            difference_vector=UtilityObject.calculateDifferenceVector(input_1,input_2);

            %calculate the l2 error
            l2_error=sqrt(sum(difference_vector.^2));        

        end       
        
        function []=checkIfInputsAreOfTheSameDimension(input_1,input_2)
            
            dimensions_of_input1=size(input_1);
            dimensions_of_input2=size(input_2);
            
            if sum(dimensions_of_input1 ~= dimensions_of_input2)
               error(['dimensions_of_input_1=' num2str(dimensions_of_input1) ' is not equal to dimensions_of_input_2=' num2str(dimensions_of_input2)] )
            end
            
        end
    end %methods (Static=true)
    
    methods (Static=true, Sealed=true)
        function[]=testMyNormCdf()
           
            error_tolerance=10^-14;
            number_of_tests_to_run=2^15;
            mean_multiplier=100*rand(1);
            standard_deviation_multiplier=100*rand(1);
            
            for index_of_number_of_tests_to_run=1:number_of_tests_to_run
               
                %generate a random meam
                random_mean=mean_multiplier*(rand(1)-0.5);
                
                %generate a standard deviation
                random_standard_deviation=standard_deviation_multiplier*abs(rand(1));
                
                %generate a list of values
                test_values=rand(index_of_number_of_tests_to_run,1);
                
                %generate cdf using myNormCdf
                my_norm_cdf=UtilityObject.myNormCdf(test_values,random_mean,random_standard_deviation);
                
                %generate cdf using MATLAB's normcdf
                matlab_norm_cdf=normcdf(test_values,random_mean,random_standard_deviation);
                
                %compare two cdfs
                l2_error=UtilityObject.calculateL2Error(my_norm_cdf,matlab_norm_cdf);
                if l2_error > error_tolerance
                    random_mean
                    random_standard_deviation
                   error(['The l2 error=' num2str(l2_error) ' between myNormCDF and matlab''s norm cdf exceeds tolerance=' num2str(error_tolerance) ' for number of test values=' num2str(length(test_values)) ', mean=' num2str(random_mean) ', std dev=' num2str(random_standard_deviation) '.' ]); 
                end
                
            end
            
        end
    end %    methods (Static=true, Sealed=true)

    
end %classdef