 function []=fplotNQuantizerLabels(simul_io_data)
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
     
     figure
     number_of_functions_per_plot=n_stddev;
     q_steps = zeros(n_qintvl,number_of_functions_per_plot);
     n_bins = q_steps;
     list_of_legend_labels=cell(1,number_of_functions_per_plot);
     
     for i_stdev_rng=1:n_stddev
         std_dev=simul_io_data.qntzr_ampl_stddev_list(i_stdev_rng);
         
         for i_q_wdth=1:n_qintvl
             q_step=simul_io_data.qntzr_wdth_mltplr_list(i_q_wdth);
             
             ttl_n_bins = 0;
             cnt_n_bins = 0;
             for i_ratio=1:n_ratios
                 cs_vid_io = simul_rslt{i_ratio, i_stdev_rng,i_q_wdth};
                 ttl_n_bins = ttl_n_bins + cs_vid_io.ac_bins_avg;
                 cnt_n_bins = cnt_n_bins + 1;
             end
             q_steps(i_q_wdth, i_stdev_rng) = q_step;
             n_bins(i_q_wdth, i_stdev_rng) = ttl_n_bins/cnt_n_bins+1;
         end
         
         list_of_legend_labels{i_stdev_rng}=...
             ['x norm ampl =' num2str(std_dev)];
     end
     
     %do plotting
     plot_handle=plot(q_steps, n_bins);
     legend(list_of_legend_labels)
     xlabel('normalized quantization step',...
         'FontSize',PlotHelper.FONTSIZE)
     ylabel('number of quantizer labesl','FontSize',...
         PlotHelper.FONTSIZE)
     title('number of quantizer label ver. quantization step',...
         'FontSize',PlotHelper.FONTSIZE)
     
     PlotHelper.prettyPlot(plot_handle);
 end
 
