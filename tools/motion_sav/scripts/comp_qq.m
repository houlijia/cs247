cases = list_files('*.case');
qq_info = struct('name', cases, 'msrs',cell(size(cases)), 'qntls', cell(size(cases)));
parfor k=1:length(cases)
    case_name = cases{k};
    cd(case_name);
    
    files = list_files('*.csvid');
    proc_opts = struct('prefix', sprintf('%d) ',k));
    [msrs, qntls] = CSVidDecoder.getNormMsrs(files,proc_opts);
    qq_info(k).msrs = msrs;
    qq_info(k).qntls = qntls;
end
save('qq_info.mat','qq_info');

for k=1:length(qq_info)
    figure;
    plot(qq_info(k).qntls, qq_info(k).msrs);
    title(qq_info(k).name);
end


