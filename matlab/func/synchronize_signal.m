% ========================================================================
% SYNCHRONIZE_SIGNAL: Align two signals based on estimated TDOA
% ========================================================================
% This function synchronizes two signals (x and y) by estimating their 
% Time Difference Of Arrival (TDOA) using Generalized Cross-Correlation (GCC)
% and aligning them accordingly.
%
% Inputs:
%   x, y : Input signals (column vectors)
% 
% Outputs:
%   x_s, y_s : Synchronized signals of equal length
% ========================================================================
function [x_s, y_s] = synchronize_siganl(x,y)
if length(y) > length(x)
    y = y(1 : length(x));
elseif length(y) < length(x)
    x = x(1 : length(y));
else
end
% A TDOA estimation can be emploied!
gama            = 0.8;
L               = 1024*8;
ro              = 1;
maxtd           = 200;
mx              = gcc(x,y,gama,L,ro,maxtd);
table           = tabulate(mx);
[~,idx]  = max(table(:,2));
timdif          = table(idx); 
fprintf('#The TDOA between the reference and observation is %.3f! #\n', timdif);

if timdif < 0
    x_s = x(1 : end + timdif);
    y_s = y(1 - timdif : end);    
else
    y_s = y(1 : end - timdif);
    x_s = x(1 + timdif : end);
end
end