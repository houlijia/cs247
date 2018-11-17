function beta = restrict_beta(beta)
    for fldc = fieldnames(beta)'
        fld = fldc{1};
        if strcmp(fld,'final')
            continue;
        end
        if beta.(fld) > beta.final.(fld)
            beta.(fld) = beta.final.(fld);
        end
    end
end
