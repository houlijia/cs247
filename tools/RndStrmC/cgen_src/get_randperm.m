function [ x, xr, y, yr ] = get_randperm( cnt, imax, seed )
    yr = rng();
    rng(seed);
    xr = rng();
    x = randperm(imax, cnt);
    rng(yr)
    y = randperm(imax, cnt);
end

