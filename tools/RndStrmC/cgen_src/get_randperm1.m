function [ x, xr, y, yr ] = get_randperm1(imax, seed )
    yr = rng();
    rng(seed);
    xr = rng();
    x = randperm(imax);
    rng(yr)
    y = randperm(imax);
end
