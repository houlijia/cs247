function [ x, xr, y, yr ] = get_rand( cnt, seed )
    yr = rng();
    rng(seed);
    xr = rng();
    x = rand(1,cnt);
    rng(yr)
    y = rand(1,cnt);
end

