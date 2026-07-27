function [nums, numv] = CalcOrderMatSize(order)
% Return the number of scalar (nums) and vector (numv) basis functions
% on a triangle for a given polynomial order.
%
%   nums = (order+1)*(order+2)/2   -- scalar basis count
%   numv = order*(order+2)          -- H(curl) basis count

nums = (order + 1) * (order + 2) / 2;
numv = order * (order + 2);

end
