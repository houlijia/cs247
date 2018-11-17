function [sml] = std_cif_q_dec(output_id, do_quant)
  [mpath,~,~] = fileparts(mfilename('fullpath'));
  cd(mpath)
  cd ..
  cd ..
  cd test_data
  do_quant_str = {'exct', 'quan'};
  q_str = do_quant_str{do_quant+1};
  anls_opts = [];
  dec_opts = struct('Q_msrmnts',do_quant, 'cmpr_blk_by_blk', 0);
  files_def = '<std_cif_io.json';
  proc_opts = struct('output_id', output_id, 'dec_id', [q_str '_*'],...
    'prefix', ['<Nn>-Mr<Mr2>Q<Qm1>-' q_str ']'], 'par_cases',0, 'par_files', 0,...
    'report_blk', false);
  sml = CSVidDecoder.doSimulation(anls_opts, dec_opts, files_def, proc_opts);
end
