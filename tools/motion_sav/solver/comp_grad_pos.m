function grad_pos = comp_grad_pos(grad_x, drct, dLd)
    % This function finds the normalized position relative to the minimum
    % of a line search.  We assume that we are given a function
    %   h(x) = f(p + x*d)
    % where x is a scalar, p and d are vectors representing the current
    % position and the direction, respectively.  Using quadratic
    % approximation,
    %   h(x) = f(p) + x* (d'*g) + 0.5*x^2*(d'Ld) where L is the Hessian of
    % f at p.  Let t = (d'*g)/sqrt(d'Ld/2).  The minimum is obtained when
    %    t = (d'*g)/(2*sqrt(d'Ld/2))= (d'*g)/sqrt(2*d'Ld)
    %             
    grad_pos = -dot(grad_x, drct) / sqrt(2 * (dLd + 1e-16));
end


