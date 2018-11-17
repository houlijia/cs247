for k=1:length(qq_info)
    figure;
    plot(qq_info(k).qntls, qq_info(k).msrs);
    title(qq_info(k).name);
end
