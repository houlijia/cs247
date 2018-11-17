function [xvec, wvec_ref] = ...
    init_solver_xvec(opts, sparser, sens_mtrx, b)
    % initialize xvec
    if isscalar(opts.init) && isnumeric(opts.init)
        switch opts.init
            case 1, xvec = sens_mtrx.multTrnspVec(b);
            otherwise, xvec = zeros(sparser.n_sigvec,1);
        end
        wvec_ref = [];
    else
        xvec = opts.init;
        if opts.use_wvec_ref
            wvec_ref = struct('vec',sparser.compSprsVec(xvec),'nrm',0);
            wvec_ref.nrm = sum(abs(wvec_ref.vec));
        else
            wvec_ref = [];
        end
    end

end

