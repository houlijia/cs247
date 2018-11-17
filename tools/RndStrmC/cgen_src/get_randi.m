function [ x, xr, y, yr ] = get_randi( cnt, seed, imax )
    yr = rng();
    rng(seed);
    xr = rng();
    x = randi(imax,1,cnt);
    rng(yr)
    y = randi(imax,1,cnt);
end

