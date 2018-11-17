function xvec = init_solver_xvec(opts, sens_mtrx, b)
    % initialize xvec
    if isscalar(opts.init) && isnumeric(opts.init)
        switch opts.init
            case 1, xvec = sens_mtrx.multTrnspVec(b);
            otherwise, xvec = sens_mtrx.zeros(sens_mtrx.nCols(),1);
        end
    else
        xvec = opts.init;
    end

end

