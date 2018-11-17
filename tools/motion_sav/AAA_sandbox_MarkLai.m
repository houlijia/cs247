%area for Mark Lai to try out different functions

sandbox_number=1;

switch sandbox_number
    
    case 1
        
        counts = [99 1]; % A one occurs 99% of the time.
        len = 1000;
        seq = randsrc(1,len,[1 2; .99 .01]); % Random sequence
        code = arithenco(seq,counts);
        s = size(code) % length of code is only 8.3% of length of seq.
        
        
    otherwise
        error(['unknown sandbox_number=' num2str(sandbox_number) ])     
end