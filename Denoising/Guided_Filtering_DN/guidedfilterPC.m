function q = guidedfilterPC(I, p, Ind, eps)
%   GUIDEDFILTER   O(1) time implementation of guided filter.
%
%   - guidance image: I (should be a gray-scale/single channel image)
%   - filtering input image: p (should be a gray-scale/single channel image)
%   - local window radius: r
%   - regularization parameter: eps


average_I = boxfilterPC(I,  Ind) ;
average_p = boxfilterPC(p,  Ind) ;
average_Ip = boxfilterPC(I.*p, Ind) ;
cov_Ip = average_Ip - average_I .* average_p; % this is the covariance of (I, p) in each local patch.

average_II = boxfilterPC(I.*I, Ind) ;
var_I = average_II - average_I .* average_I;

a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;
b = average_p - a .* average_I; % Eqn. (6) in the paper;

average_a = boxfilterPC(a, Ind);
average_b = boxfilterPC(b, Ind);

q = a .* average_I + b; % Eqn. (8) in the paper;
end