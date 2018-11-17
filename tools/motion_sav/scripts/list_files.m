function files = list_files(spec)
    ds = dir(spec);
    files = struct2cell(ds);
    files = files(find(strcmp(fieldnames(ds),'name'),1),:);
end