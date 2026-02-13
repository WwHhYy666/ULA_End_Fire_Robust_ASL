function sn = STFTInv(sf, tfArg)
%   F*T*M------>T*M 
%
%   Title:
%       Inverse Short-time Fourier transform (STFT) for multichannel inputs.
%
%   Usage:
%       sn = STFTInv(sf);
%       sn = STFTInv(sf, tfArg);
%
%   Inputs:
%       sf: signal in STFT domain (fBin * nFrm * nCh);
%       tfArg:
%           -frm: frame length;
%           -inc: increment;
%           -win: analysis window;
%
%   Outputs:
%       sn: signal in time domain (samples * nCh);
%
%   Reference:
%       D. Griffin and J. Lim, "Signal Estimation from Modified Short-Time Fourier Transform", in
%   IEEE Trans. Audio, Speech, Lang. Process., vol. 32, no. 2, pp. 236-243, 1984.
%
%   Author:
%       Jilu Jin, Center of Intelligent Acoustics and Immersive Communications,
%       charles.jilu.jin @ ieee.org

%% check inputs

narginchk(1, 2);

if nargin < 2
    frm = 2^9;
    inc = 2^7;
    kwin = kaiser(frm, 1.9*pi);
    pkw = sqrt(sum(kwin.^2)/inc);
    win = kwin(:) / pkw;
else
    try
        frm = tfArg.frm;
        inc = tfArg.inc;
        win = tfArg.win(:);
    catch
        error('Invalide Arguments!')
    end
end

%% inverse STFT process
[nBin, nFrm, nCh] = size(sf);
win = optWin(win, inc);
sLen = (nFrm + ceil(frm/inc) - 1) * inc;

sn = zeros(sLen, nCh);
sf = [sf; conj(flipud(sf(2:nBin-1, :, :)))];
sFrm = real(ifft(sf)) .* win;
for frmIdx = 1:nFrm
    idx = (frmIdx-1)*inc+1:(frmIdx-1)*inc+frm;
    sn(idx, :) = sn(idx, :) + squeeze(sFrm(:, frmIdx, :));
end
end

function synWin = optWin(anaWin, inc)
    winMat = reshape(anaWin, inc, []);
    winMat = winMat ./ sum(winMat.^2, 2);
    synWin = winMat(:);
end