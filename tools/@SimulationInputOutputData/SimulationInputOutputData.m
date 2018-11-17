classdef SimulationInputOutputData < handle
    % This object contains input and output data for a specific simulation, a
    % series of test. A specific test is uniquely defined by three values
    % --> the ratio_of_number_of_cs_measurements_to_input_values was used
    % --> the number of standard deviaions preserved
    % --> the quantization step size (specificied by a multiple of the
    % "normalized" quantization step size)
    
    
    properties
        
        %which directory the simulation output data was written to
        full_path_of_output;
        
        %a list of ratio of number of cs measurements to input values that
        %was tested
        msrmnt_input_ratio_list;
        
        %a list of quantizatio interval widht multiplier that was tested
        qntzr_wdth_mltplr_list;
        
        %a list of standard deviations to preserve that was tested
        qntzr_ampl_stddev_list;
        
        % This is a 3-D cell array.
        % The first dimension contains indicies of msrmnt_input_ratio_list.
        % The second dimension contains indices of qntzr_ampl_stddev_list.
        % The third dimension conatins indices of qntzr_wdth_mltplr_list.
        % Each entry is an object of type CSVideoCodecInputOutputData containing
        % the simulation results
        simulation_results;
        
        %the number of seconds to complete the entire simulation
        total_seconds_to_do_simulation;
    end
    
    properties (Constant)
        line_props = {'-xb', '-ok', '-dg', '-+m', '-^c', '-sk', '-vr'};
    end
        
    methods
        function setResults(obj, simul_data)
            n_cases = length(simul_data);
            obj.msrmnt_input_ratio_list = zeros(1,n_cases);
            obj.qntzr_ampl_stddev_list = zeros(1,n_cases);
            obj.qntzr_wdth_mltplr_list = zeros(1,n_cases);
            for k = 1:n_cases
                obj.msrmnt_input_ratio_list(k) = ...
                    simul_data{k}.msrmnt_input_ratio;
                obj.qntzr_ampl_stddev_list(k) = ...
                    simul_data{k}.qntzr_ampl_stddev;
                obj.qntzr_wdth_mltplr_list(k) = ...
                    simul_data{k}.qntzr_wdth_mltplr;
            end
            obj.msrmnt_input_ratio_list = ...
                unique(obj.msrmnt_input_ratio_list);
            obj.qntzr_ampl_stddev_list = ...
                unique(obj.qntzr_ampl_stddev_list);
            obj.qntzr_wdth_mltplr_list = ...
                unique(obj.qntzr_wdth_mltplr_list);
            
            obj.simulation_results = cell(...
                length(obj.msrmnt_input_ratio_list),...
                length(obj.qntzr_ampl_stddev_list),...
                length(obj.qntzr_wdth_mltplr_list));
            for k=1:length(simul_data)
                indx_ratio = find(obj.msrmnt_input_ratio_list == ...
                    simul_data{k}.msrmnt_input_ratio,1);
                indx_ampl_stddev = find(...
                    obj.qntzr_ampl_stddev_list == ...
                    simul_data{k}.qntzr_ampl_stddev, 1);
                indx_wdth_mltplr = find(...
                    obj.qntzr_wdth_mltplr_list == ...
                    simul_data{k}.qntzr_wdth_mltplr, 1);
                
                obj.simulation_results{indx_ratio,...
                    indx_ampl_stddev, indx_wdth_mltplr} = ...
                    simul_data{k};
            end
        end
        
        function [n_ratios, n_stddev, n_qintvl] = getDims(simul_io_data)
            n_ratios=size(simul_io_data.simulation_results,1);
            n_stddev=size(simul_io_data.simulation_results,2);
            n_qintvl=size(simul_io_data.simulation_results,3);
        end
        
        function plot_all(simul_io_data, sub_clp_rate)
            % this is a wrapper function that calls the other plotting functions
            
            if nargin < 2
                sub_clp_rate = false;
            end
            
            simul_io_data.plotFrctMsrsClipped();
            simul_io_data.plotBitsPerMeasurementVsQuantizationStep();
            %plotNQuantizerLabels(simul_io_data);
            simul_io_data.plotPsnrVsBitRateForEachQuantizationWidth(sub_clp_rate);
            simul_io_data.plotPsnrVsBitRateForEachStdDevPreserved(sub_clp_rate);
        end
        
        function plotPsnrVsBitRateForEachStdDevPreserved(simul_io_data, ...
                sub_clp_rate)
            %plots PSNR vs Bit Rate.  Each figure conatins the PSNR/Bit Rate graph
            %for a fixed "standard_deviation_to_preserve" (determines range of 
            %qunatizer) value.
            %
            % [Input]
            %
            % simul_io_data - a SimulationInputOutputData object
            % sub_clp_rate - if present and true, subtract from the bit
            %   rates the bits spent on indicating if clipped or not and use
            %   the reduced bit rate. If eq. 2 draw rate-PSNR curves with
            %   both (the one with reduced bit rate is dotted).
            %
            % [Output]
            %
            % <none>
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            if nargin < 2
                sub_clp_rate = false;
            end
            
            [~, n_stddev, n_qintvl] = simul_io_data.getDims();
            [bit_rates, psnrs, ~, entrp_clp_br, ~] = ...
                simul_io_data.populateTables();
            
            bit_rates_sub = bit_rates - entrp_clp_br;
            
            % Draw the figures
            axes_range = [floor(min(min(min(bit_rates_sub*0.95))))...
                ceil(max(max(max(bit_rates*1.05)))) ...
                    floor(min(min(min(psnrs))))-1 ceil(max(max(max(psnrs))))+1];

            for i_stdev_rng=1:n_stddev
                %plot a PSNR/Bit Rate for each standard deviation preserverd
                figure
                
                %a cell row of strings that will be used to generate the legend
                if sub_clp_rate == 2
                    list_of_legend_label=...
                        cell(1,2*n_qintvl);
                else
                    list_of_legend_label=...
                        cell(1,n_qintvl);
                end
                
                %plot it
                n_props = length(SimulationInputOutputData.line_props);
                hold on;
                if sub_clp_rate == 0 || sub_clp_rate == 2
                    for i_q_wdth=1:n_qintvl
                        plot_handle=plot(bit_rates(:,i_q_wdth,i_stdev_rng),...
                            psnrs(:,i_q_wdth,i_stdev_rng),...
                            SimulationInputOutputData.line_props{1+mod(i_q_wdth,n_props)});
                        list_of_legend_label{i_q_wdth}=...
                            sprintf('NrmStp: %5.1f', ...
                            simul_io_data.qntzr_wdth_mltplr_list(i_q_wdth));
                    end
                    bgn = n_qintvl;
                else
                    bgn = 0;
                end
                if sub_clp_rate
                    for i_q_wdth=1:n_qintvl
                        l_prop = SimulationInputOutputData.line_props{...
                            1+mod(i_q_wdth,n_props)};
                        l_prop = [':' l_prop(2:end)];
                        plot_handle=plot(bit_rates_sub(:,i_q_wdth,i_stdev_rng),...
                            psnrs(:,i_q_wdth,i_stdev_rng),l_prop);
                        list_of_legend_label{i_q_wdth+bgn}=...
                            sprintf('NrmStp: %5.1f (NoSat)', ...
                            simul_io_data.qntzr_wdth_mltplr_list(i_q_wdth));
                    end
                end
                hold off;
                axis(axes_range);
                
                legend(list_of_legend_label);
                xlabel('Bit Rate (bits/second)','FontSize',PlotHelper.FONTSIZE)
                ylabel('PSNR','FontSize',PlotHelper.FONTSIZE)
                title(sprintf('Rate distortion functions for Nrm.Ampl=%3.1f',...
                    simul_io_data.qntzr_ampl_stddev_list(i_stdev_rng)),...
                    'FontSize',PlotHelper.FONTSIZE)
                
                PlotHelper.prettyPlot(plot_handle);
                
            end
            
            
        end
        
        function []=plotPsnrVsBitRateForEachQuantizationWidth(simul_io_data ,...
                sub_clp_rate)
            %plots PSNR vs Bit Rate.  Each figure conatins the PSNR/Bit Rate 
            %graph for a fixed quantization step width (specified as a multiple 
            %of the "normalized" quantization step) value.
            %
            % [Input]
            %
            % simul_io_data - a SimulationInputOutputData object
            % sub_clp_rate - if present and true, subtract from the bit
            %   rates the bits spent on indicating if clipped or not and use
            %   the reduced bit rate. If eq. 2 draw rate-PSNR curves with
            %   both (the one with reduced bit rate is dotted).
            %
            % [Output]
            %
            % <none>
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            if nargin < 2
                sub_clp_rate = false;
            end
            
            [~, n_stddev, n_qintvl] = simul_io_data.getDims();
            
            [bit_rates, psnrs, ~, entrp_clp_br, ~] = ...
                simul_io_data.populateTables();
            
            bit_rates_sub = bit_rates - entrp_clp_br;
            
            % Draw the figures
            axes_range = [floor(min(min(min(bit_rates_sub*0.95)))) ...
                ceil(max(max(max(bit_rates*1.05)))) ...
                    floor(min(min(min(psnrs))))-1 ceil(max(max(max(psnrs))))+1];

            for i_q_wdth=1:n_qintvl
                %plot a PSNR/Bit Rate for each quantization_interval_width
                figure
                
                %a cell row of strings that will be used to generate the legend
                if sub_clp_rate == 2
                    list_of_legend_label=cell(1,n_stddev*2);
                else
                    list_of_legend_label=cell(1,n_stddev);
                end
                
                %plot it
                n_props = length(SimulationInputOutputData.line_props);
                hold on;
                if sub_clp_rate == 0 || sub_clp_rate == 2
                    for i_stddev_rng=1:n_stddev
                        plot_handle=plot(bit_rates(:,i_q_wdth,i_stddev_rng),...
                            psnrs(:,i_q_wdth,i_stddev_rng),...
                            SimulationInputOutputData.line_props{1+mod(i_stddev_rng,n_props)});
                        list_of_legend_label{i_stddev_rng}=...
                            sprintf('NrmAmpl: %5.1f', ...
                            simul_io_data.qntzr_ampl_stddev_list(i_stddev_rng));
                    end
                    bgn = n_stddev;
                else
                    bgn = 0;
                end
                if sub_clp_rate
                    for i_stddev_rng=1:n_stddev
                        l_prop = SimulationInputOutputData.line_props{...
                            1+mod(i_stddev_rng,n_props)};
                        l_prop = [':' l_prop(2:end)];
                        plot_handle=plot(bit_rates_sub(:,i_q_wdth,i_stddev_rng),...
                            psnrs(:,i_q_wdth,i_stddev_rng), l_prop);
                        list_of_legend_label{i_stddev_rng+bgn}=...
                            sprintf('NrmAmpl: %5.1f (NoSat)', ...
                            simul_io_data.qntzr_ampl_stddev_list(i_stddev_rng));
                    end
                end
                hold off;
                axis(axes_range);
                
                legend(list_of_legend_label);
                xlabel('Bit Rate (bits/second)','FontSize',PlotHelper.FONTSIZE)
                ylabel('PSNR','FontSize',PlotHelper.FONTSIZE)
                
                title(sprintf('Rate distortion functions for Nrm.Step=%3.1f',...
                    simul_io_data.qntzr_wdth_mltplr_list(i_q_wdth)),...
                    'FontSize',PlotHelper.FONTSIZE)
                
                PlotHelper.prettyPlot(plot_handle);
            end
            
        end
        
        function []=plotBitsPerMeasurementVsQuantizationStep(simul_io_data)
            %plots Bits Per Compressed Sensed Measurment and Bits Per Used 
            %Compressed Sensed Measurement vs Quantization Step. Since some 
            %of the Compressed Sensed Measurement are discarded because they
            %lie outside the range of the quantizer,the two plots are not identical.
            %
            %Each figure contains the graph for a fixed ratio of number of
            %compressed sensed measurements to input values.
            %
            % [Input]
            %
            % simul_io_data - a SimulationInputOutputData object
            %
            % [Output]
            %
            % <none>
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %extract the simulation results
            simul_rslt=simul_io_data.simulation_results;
            
            %calculate the number of tests run for each variable of the simulation
            %  -->ratio of compressed sensed measurements to input values
            %  -->standard deviations preserved by quantizer
            %  -->quantiztion interval width multiplier
            [n_ratios, n_stddev, n_qintvl] = simul_io_data.getDims();
            
            number_of_functions_per_plot=2*n_stddev;
            q_steps_per_stddev=zeros(n_qintvl,n_stddev);
            bits_per_msr = q_steps_per_stddev;
            bits_per_used_msr  = q_steps_per_stddev;
            clp_ovhd = q_steps_per_stddev;
            fxd_ovhd = q_steps_per_stddev;
            msr_ovhd = q_steps_per_stddev;
            legend_labels=cell(1,number_of_functions_per_plot);
            entrp_clp = zeros(n_ratios, n_stddev, n_qintvl);
            msrmnt_bits = entrp_clp;
            stdv = entrp_clp;
            wdth = entrp_clp;
                
            for i_stdev_rng=1:n_stddev
                for i_q_wdth=1:n_qintvl
                    
                    %pull out raw data
                    q_steps=simul_io_data.qntzr_wdth_mltplr_list(i_q_wdth);
                    std_devs=simul_io_data.qntzr_ampl_stddev_list(i_stdev_rng);
                    
                    ttl_msrmnts_bits = 0;
                    ttl_msrmnts = 0;
                    ttl_msrmnts_outsd = 0;
                    ttl_entrp_clp = 0;
                    ttl_entrp_bits = 0;
                    ttl_fxd_bits = 0;
                    for i_ratio=1:n_ratios
                        cs_vid_io = simul_rslt{i_ratio, i_stdev_rng,i_q_wdth};
                        ttl_msrmnts_bits = ttl_msrmnts_bits + ...
                            cs_vid_io.ttl_msrmnts_bits;
                        ttl_msrmnts = ttl_msrmnts + cs_vid_io.ttl_msrmnts;
                        ttl_msrmnts_outsd = ttl_msrmnts_outsd + ...
                            cs_vid_io.ttl_msrmnts_outsd;
                        ttl_entrp_bits = ttl_entrp_bits + ...
                            cs_vid_io.ttl_entrp_bits;
                        ttl_entrp_clp = ttl_entrp_clp + ...
                            cs_vid_io.ttl_entrp_clp;
                        ttl_fxd_bits= ttl_fxd_bits + ...
                            cs_vid_io.ttl_fxd_bits;
                        entrp_clp(i_ratio, i_stdev_rng, i_q_wdth) = cs_vid_io.ttl_entrp_bits;
                        msrmnt_bits(i_ratio, i_stdev_rng, i_q_wdth) = cs_vid_io.ttl_msrmnts_bits;
                        stdv(i_ratio, i_stdev_rng, i_q_wdth) = cs_vid_io.qntzr_ampl_stddev;
                        wdth(i_ratio, i_stdev_rng, i_q_wdth) = cs_vid_io.qntzr_wdth_mltplr;
                    end
                    
                    %do calculations and store values
                    q_steps_per_stddev(i_q_wdth,i_stdev_rng)=q_steps;
                    
                    bits_per_msr(i_q_wdth,i_stdev_rng)=...
                        ttl_msrmnts_bits / ttl_msrmnts;
                    bits_per_used_msr(i_q_wdth,i_stdev_rng)=ttl_msrmnts_bits / ...
                        (ttl_msrmnts - ttl_msrmnts_outsd);
                    clp_ovhd(i_q_wdth, i_stdev_rng) = ...
                        ttl_entrp_clp / ttl_msrmnts_bits;
                    msr_ovhd(i_q_wdth, i_stdev_rng) = ...
                        ttl_msrmnts_bits / ttl_entrp_bits - 1;
                    fxd_ovhd(i_q_wdth, i_stdev_rng) = ...
                        1 - ttl_msrmnts_bits/ttl_fxd_bits ;
                    
                end
                legend_labels{i_stdev_rng}=...
                    ['ampl.=' num2str(std_devs)];
                legend_labels{i_stdev_rng+n_stddev}=...
                    ['bit/used msr, ampl.=' num2str(std_devs)];
                    
            end
            
            % sort
            [q_steps_per_stddev, ordr] = sort(q_steps_per_stddev);
            for k=1:size(ordr,2)
                bits_per_msr(:,k) = bits_per_msr(ordr(:,k),k);
                bits_per_used_msr(:,k) = bits_per_used_msr(ordr(:,k),k);
                clp_ovhd(:,k) = clp_ovhd(ordr(:,k),k);
                msr_ovhd(:,k) = msr_ovhd(ordr(:,k),k);
                fxd_ovhd(:,k) = fxd_ovhd(ordr(:,k),k);
            end
            
            %do plotting
            figure
            plot_handle=plot([q_steps_per_stddev q_steps_per_stddev],...
                [bits_per_msr bits_per_used_msr]);
            legend(legend_labels)
            xlabel('Normalized quantization step','FontSize',PlotHelper.FONTSIZE)
            ylabel('bits per measurement','FontSize',...
                PlotHelper.FONTSIZE)
            title('bits per measurement versus quantization step ',...
                'FontSize', PlotHelper.FONTSIZE )
            
            PlotHelper.prettyPlot(plot_handle);

            figure
            plot_handle=plot(q_steps_per_stddev, clp_ovhd);
            legend(legend_labels(1:size(q_steps_per_stddev,2)))
            xlabel('Normalized quantization step','FontSize',PlotHelper.FONTSIZE)
            ylabel('Fraction','FontSize',...
                PlotHelper.FONTSIZE)
            title('Fraction of measurements bits used to indicate saturation',...
                'FontSize', PlotHelper.FONTSIZE )
            
            PlotHelper.prettyPlot(plot_handle);

            figure
            plot_handle=plot(q_steps_per_stddev, msr_ovhd);
            legend(legend_labels(1:size(q_steps_per_stddev,2)))
            xlabel('Normalized quantization step','FontSize',PlotHelper.FONTSIZE)
            ylabel('Fraction','FontSize',...
                PlotHelper.FONTSIZE)
            title('Increase in bit rate per measurement relative to entropy',...
                'FontSize', PlotHelper.FONTSIZE)
            
            PlotHelper.prettyPlot(plot_handle);

            figure
            plot_handle=plot(q_steps_per_stddev, fxd_ovhd);
            legend(legend_labels(1:size(q_steps_per_stddev,2)))
            xlabel('Normalized quantization step','FontSize',PlotHelper.FONTSIZE)
            ylabel('Fraction','FontSize',...
                PlotHelper.FONTSIZE)
            title('Saving relative to fixed size coding', 'FontSize', PlotHelper.FONTSIZE )
            
            PlotHelper.prettyPlot(plot_handle);
        end
        
        function []=plotFrctMsrsClipped(simul_io_data)
            % plots fraction of measurements outside the range of quantizer 
            % range versus the number of standard deviations preserved by the
            % quantizer
            %
            % Each figure contains the graph for a fixed ratio of number of
            % compressed sensed measurements to input values.
            %
            % [Input]
            %
            % simul_io_data - a SimulationInputOutputData object
            %
            % [Output]
            %
            % <none>
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %calculate the number of tests run for each variable of the simulation
            %  -->ratio of compressed sensed measurements to input values
            %  -->standard deviations preserved by quantizer
            %  -->quantiztion interval width multiplier
            [n_ratios, n_stddev, n_qintvl] = simul_io_data.getDims();
            simul_rslt=simul_io_data.simulation_results;

            figure
            number_of_functions_per_plot=n_qintvl;
            std_devs=zeros(n_stddev,number_of_functions_per_plot);
            frac_outside_per_q_step=std_devs;
            legend_labels=cell(1,number_of_functions_per_plot);
                
            for i_q_wdth=1:n_qintvl
                for i_stdev_rng=1:n_stddev
                    
                    %pull out raw data
                    q_step=simul_io_data.qntzr_wdth_mltplr_list(i_q_wdth);
                    std_dev=simul_io_data.qntzr_ampl_stddev_list(i_stdev_rng);
                    
                    %do calculations
                    ttl_msrmnts = 0;
                    ttl_msrmnts_outsd = 0;
                    for i_ratio=1:n_ratios
                        cs_vid_io = simul_rslt{i_ratio, i_stdev_rng,i_q_wdth};
                        ttl_msrmnts = ttl_msrmnts + cs_vid_io.ttl_msrmnts;
                        ttl_msrmnts_outsd = ttl_msrmnts_outsd + ...
                            cs_vid_io.ttl_msrmnts_outsd;
                    end
                    
                    %store values
                    std_devs(i_stdev_rng,i_q_wdth)=std_dev;
                    
                    frac_outside_per_q_step(i_stdev_rng,i_q_wdth)= ...
                        ttl_msrmnts_outsd /  ttl_msrmnts;
                    
                    legend_labels{i_q_wdth}=...
                        sprintf('nrm Q step=%5.1f',q_step);
                    
                end
            end
            
            %do plotting
            plot_handle=plot(std_devs,frac_outside_per_q_step);
               
            legend(legend_labels)
            xlabel('Normalized quantizer amplitude','FontSize',PlotHelper.FONTSIZE)
            ylabel('fraction of discarded measurements','FontSize',...
                PlotHelper.FONTSIZE)
            title('Discarded measurements ver. quantizer amplitude',...
                'FontSize',PlotHelper.FONTSIZE)
            
            PlotHelper.prettyPlot(plot_handle);
        end
        
        function []=plotNQuantizerLabels(simul_io_data)
            % plots the average number of quantizer labels versus normalized
            % quantization steps, differnt values of quantizer amplitude.
            %
            % simul_io_data - a SimulationInputOutputData object
            %
            % [Output]
            %
            % <none>
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            SimulationInputOutputData.splotNQuantizerLabels(simul_io_data);
        end
                    
        function [bit_rates, psnrs, entrp_rates, fixed_rates, clp_rates] = ...
                populateTables(simul_io_data)
            
            [n_ratios, n_stddev, n_qintvl] = simul_io_data.getDims();
            
            bit_rates=zeros(n_ratios,n_qintvl, n_stddev);
            psnrs=bit_rates;
            entrp_rates = bit_rates;
            fixed_rates = bit_rates;
            clp_rates = bit_rates;
            
            % Populate the tables
            for i_stdev_rng=1:n_stddev
               for i_ratio=1:n_ratios
                     for i_q_wdth=1:n_qintvl
                        %pull out raw data
                        cs_vid_io = simul_io_data.simulation_results{i_ratio,...
                            i_stdev_rng,i_q_wdth};
                        
                        %populate the lists
                        [bit_rates(i_ratio,i_q_wdth, i_stdev_rng),...
                            entrp_rates(i_ratio,i_q_wdth, i_stdev_rng),...
                            fixed_rates(i_ratio,i_q_wdth, i_stdev_rng)] =...
                            cs_vid_io.bitRate();
                        psnrs(i_ratio,i_q_wdth, i_stdev_rng)=cs_vid_io.psnr;
                     end
               end
            end
        end
        
    end  % Methods
    
    methods (Static)
        function []=splotNQuantizerLabels(simul_io_data)
            % plots the average number of quantizer labels versus normalized
            % quantization steps, differnt values of quantizer amplitude.
            %
            % simul_io_data - a SimulationInputOutputData object
            %
            % [Output]
            %
            % <none>
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %calculate the number of tests run for each variable of the simulation
            %  -->ratio of compressed sensed measurements to input values
            %  -->standard deviations preserved by quantizer
            %  -->quantiztion interval width multiplier
            [n_ratios, n_stddev, n_qintvl] = simul_io_data.getDims();
            simul_rslt=simul_io_data.simulation_results;

            number_of_functions_per_plot=n_stddev;
            q_steps = zeros(n_qintvl,number_of_functions_per_plot);
            n_bins = q_steps;
            max_bins = q_steps;
            min_bins = q_steps;
            legend_labels=cell(1,number_of_functions_per_plot);
                
            for i_stdev_rng=1:n_stddev
                std_dev=simul_io_data.qntzr_ampl_stddev_list(i_stdev_rng);

                for i_q_wdth=1:n_qintvl
                    q_step=simul_io_data.qntzr_wdth_mltplr_list(i_q_wdth);
                    
                    ttl_n_bins = 0;
                    cnt_n_bins = 0;
                    max_n_bins = 0;
                    min_n_bins = inf;
                    for i_ratio=1:n_ratios
                        cs_vid_io = simul_rslt{i_ratio, i_stdev_rng,i_q_wdth};
                        ttl_n_bins = ttl_n_bins + cs_vid_io.bins_avg;
                        cnt_n_bins = cnt_n_bins + 1;
                        max_n_bins = max(max_n_bins, cs_vid_io.bins_max);
                        min_n_bins = min(min_n_bins, cs_vid_io.bins_min);
                    end
                    q_steps(i_q_wdth, i_stdev_rng) = q_step;
                    n_bins(i_q_wdth, i_stdev_rng) = ttl_n_bins/cnt_n_bins + 1;
                    max_bins(i_q_wdth, i_stdev_rng) = max_n_bins;
                    min_bins(i_q_wdth, i_stdev_rng) = min_n_bins;
                end
                
                legend_labels{i_stdev_rng}=...
                    ['x norm ampl =' num2str(std_dev)];
            end
                
            %do plotting
            n_props = length(SimulationInputOutputData.line_props);
            figure
            hold on
            for k = 1:size(q_steps,2)
                l_prop = SimulationInputOutputData.line_props{1+mod(k,n_props)};
                plot_handle=plot(q_steps(:,k), n_bins(:,k), l_prop);
            end
            legend(legend_labels)
            xlabel('normalized quantization step',...
                'FontSize',PlotHelper.FONTSIZE)
            ylabel('number of quantizer labesl','FontSize',...
                PlotHelper.FONTSIZE)
            title('number of quantizer label ver. quantization step',...
                'FontSize',PlotHelper.FONTSIZE)
            for k = 1:size(q_steps,2)
                l_prop = SimulationInputOutputData.line_props{1+mod(k,n_props)};
                l_prop = [':' l_prop(2:end)];
                plot_handle=plot(q_steps(:,k), min_bins(:,k), l_prop);
            end
            for k = 1:size(q_steps,2)
                l_prop = SimulationInputOutputData.line_props{1+mod(k,n_props)};
                l_prop = ['-.' l_prop(2:end)];
                plot_handle=plot(q_steps(:,k), max_bins(:,k), l_prop);
            end
           
%             for k = 1:size(q_steps,2)
%                 l_prop = SimulationInputOutputData.line_props{1+mod(k,n_props)};
%                 for j=1:size(q_steps,1)
%                     plot_handle=plot([q_steps(j,k); q_steps(j,k)],...
%                         [max_bins(j,k);min_bins(j,k)], l_prop);
%                 end                        
%             end
            hold off
            PlotHelper.prettyPlot(plot_handle);
        end
                    
    end
    
end