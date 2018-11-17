classdef PlotHelper

    properties (Constant=true)
        DATA_LINEWIDTH=1.8;
        LINESTYLE='--'; % '-' '--' ':'  '-.''none' --> type "help plot" for details
        MARKER='x'; %'.','o','x','+','*','s','d','v','^','<','>','>','p','h' --> type "help plot" for details
%        MARKERSIZE=15;
        MARKERSIZE=10;
        
        FONTSIZE=15;
        AXIS_LINEWIDTH=0.5;
    end

    methods (Static=true)
      
       function []=prettyPlot(figure_handle)
            %PRETTYPLOT makes the figure look nice for publishing
            %@param figure_handle - the handle to the figure that is to be updated
            %@return void - nothing, the figure should be updated

                set(figure_handle     , ...
                  ...'Color'           , PlotHelper.getRGBColor('dark_slate_gray') , ... %color of the lines
                  'LineWidth'       , PlotHelper.DATA_LINEWIDTH , ...
                  'LineStyle'       , PlotHelper.LINESTYLE, ... 
                  ...'Marker'          , PlotHelper.MARKER, ...        
                  ...'MarkerEdgeColor' , PlotHelper.getRGBColor('midnight_blue') , ...
                  ...'MarkerFaceColor' , PlotHelper.getRGBColor('midnight_blue')      , ...
                  'MarkerSize'      , PlotHelper.MARKERSIZE          )

              %update the axis
                set(gca, ...
                  'Box'         , 'off'     , ...
                  'TickDir'     , 'out'     , ...
                  'TickLength'  , [.01 .01] , ...
                  'XMinorTick'  , 'off'      , ...
                  'YMinorTick'  , 'off'      , ...
                  'XGrid'       , 'off'     , ...
                  'YGrid'       , 'off'      , ...
                  'XColor'      , [.3 .3 .3], ...
                  'YColor'      , [.3 .3 .3], ...
                  'FontSize'    , PlotHelper.FONTSIZE    , ...
                  'FontWeight'  , 'bold', ...
                  'LineWidth'   , PlotHelper.AXIS_LINEWIDTH        );

        end
        
        function [ color_string ]=getColorFromPalette ( palette_bin_number)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % getColorFromPalette gets color from palette based on the bin number, if
        % the bin number is out of range, the mod of the bin number by the total number of bins is taken
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%INPUT PARAMETERS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % palette_bin_number - a positive integer specifying the bin number of
        % palette, each bin number correpsonds to a color, not necessarily unique
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%RETURN VALUES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % color_string- a string specifying the color name chosen from the palette
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %define the palette of colors 
        palette_of_color_strings={ ...
            'lime_green' , ...
            'dodger_blue', ...
            'orange_red', ...
            'purple', ...
            'deep_pink', ...
            'firebrick', ...
            'forest_green', ...
            'cornflower_blue'};

        %determine total number of palette bins
        num_of_palette_bins=length(palette_of_color_strings);

        %mod the palette_bin_number by the total number of palette bins to insure a
        % a bin is returned
        mod_palette_bin_number=mod(palette_bin_number,num_of_palette_bins);

        %get a color from the palette
        color_string=palette_of_color_strings{mod_palette_bin_number+1};

        end    

        function [ rgb_color_row ]= getRGBColor( rgb_color_string)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % GETRGBCOLOR returns the rgb color, a  3 entry with values between 0 and 1,
        % specified by the user, the values are found from this website
        % http://www.tayloredmktg.com/rgb/
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%INPUT PARAMETERS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % rgb_color_string - a string specifying the colro name
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%RETURN VALUES%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % rgb_color_row - a 3 entry row specifying r,g,b values, each value
        %   is betweeen 0 and 1
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %max levels
        max_level=255;

        switch rgb_color_string
            case 'medium_purple'
                %Medium Purple 	147-112-219
                 rgb_color_row=[147 112 219];        

            case 'olive_drab'
                %olive_drab 	107-142-35
                rgb_color_row=[107 142 35];        

            case 'lime_green'
                %lime green 	50-205-50
                rgb_color_row=[50 205 50];        

            case 'dark_slate_gray'
                %Dark Slate Gray  	49-79-79
                rgb_color_row=[49 79 79];

            case 'midnight_blue'
                %Midnight Blue  	25-25-112
                rgb_color_row=[25 25 112];

            case 'dodger_blue'
                %Dodger Blue  	30-144-255
                rgb_color_row=[30 144 255];

            case 'orange_red'
                %Orange Red  	255-69-0
                rgb_color_row=[255 69 0];

            case 'medium_slate_blue'
                %Medium Slate Blue  	123-104-238
                 rgb_color_row=[123 104 238];

            case 'purple'
                %Purple  	160-32-240
                rgb_color_row=[160 32 240];

            case 'deep_pink'
                %Deep Pink  	255-20-147
                rgb_color_row=[255 20 147];

            case 'firebrick'
                %Firebrick  	178-34-34
                rgb_color_row=[178 34 34];

            case 'forest_green'
                %Forest Green  	34-139-34
                rgb_color_row=[34 139 34];

            case 'cornflower_blue'
                %Cornflower Blue  	100-149-237
                rgb_color_row=[100 149 237];

            otherwise
                error(['The rgb_color_string=' rgb_color_string ' is not supported'])

        end

        %normalize the rgb_color_row to be between 0 and 1
        rgb_color_row=rgb_color_row./max_level;

        end    
      
        
        
       
    end %methods (Static=true)

end