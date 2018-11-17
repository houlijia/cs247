function [ x, xr, y, yr ] = get_randn( cnt, seed )
    yr = rng();
    rng(seed);
    xr = rng();
    x = randn(1,cnt);
    rng(yr)
    y = randn(1,cnt);
end

