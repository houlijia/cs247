cases = list_files('*.case');
base=pwd;
axes_size = [-5.5, 5.5, -5.5, 5.5];

for k=1:length(cases)
    case_name = cases{k};
    name = regexprep(case_name,'.case$','');
    cd(case_name);
    h = figure;
    axis(axes_size);
    title(name);
    
    files = list_files('*.csvid');
    proc_opts = struct('prefix', sprintf('%d) ',k), ...
        'blk_rng', [1,1,1;inf,inf,inf], 'title', case_name);
    CSVidDecoder.getNormBlkMsrs(files, h, proc_opts);
    cd(base);
    saveas(h, name, 'fig');
    close(h);
end
