function [sf,stft_parameters] = STFT(sn, tfArg)
%   T*M--->F*T*M
%
%   Title:
%       Short-time Fourier transform (STFT) for multichannel inputs.
%
%   Usage:
%       sf = STFT(sn);
%       sf = STFT(sn, tfArg);
%
%   Inputs:
%       sn: signal in time domain (samples * channels);
%       tfArg:
%           -frm: frame length;
%           -inc: increment; equal to hop_size
%           -win: analysis window;
%
%   Outputs:
%       sf: signal in STFT domain (fBin * nFrm * nCh);
%
%   Author:
%       Jilu Jin, Center of Intelligent Acoustics and Immersive Communications,
%       charles.jilu.jin @ ieee.org

%% check inputs

narginchk(1, 2);

if nargin < 2
    frm = 2^9;% 512
    inc = 2^7;% 128
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

%% STFT process

[L, nCh] = size(sn);
nFrm = ceil(L/inc) - floor(frm/inc) + 1;
sf = zeros(frm, nFrm, nCh);

for mIdx = 1:nCh
    sf(:, :, mIdx) = buffer(sn(:, mIdx), frm, frm-inc, 'nodelay');
end

sf = fft(sf .* win);
sf = sf(1:frm/2+1, :, :);

% save these parameters for istft
stft_parameters.frmsize     = inc;
stft_parameters.winsize     = frm;
stft_parameters.win         = win;
stft_parameters.nfbands     = frm/2+1;
stft_parameters.nfrms       = nFrm;

end