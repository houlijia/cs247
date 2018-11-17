classdef CSVInFile < TextInFile & CSVIn
    %CSVIn where the input is from a file
    
    properties
    end
    
    methods
        function obj = CSVInFile(fd)
            obj = obj@TextInFile(fd);
            obj = obj@CSVIn();
        end
    end
    
end

