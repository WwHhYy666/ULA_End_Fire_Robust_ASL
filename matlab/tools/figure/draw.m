% ========================================================================
% DRAW: Generate DOA Estimation Performance Plots
% ========================================================================
% This script loads DOA estimation results and generates three plots:
% - RMSE (Root Mean Square Error) vs. angle
% - Soft-Accuracy (S-ACC) vs. angle
% - Degradation span visualization
% All four estimation methods are compared (MVDR, SRP-PHAT, W-SRP-PHAT, GCC-WLS)
% ========================================================================

clear; close all; clc;

% ===== Loading Configuration =====
% Specify input MAT file containing DOA estimation results
mat_file_all = 'C:\Users\25401\Desktop\End-Fire\result\DOA_results_2m.mat';

% Method column names in the results table
method_cols = {'Est_MVDR','Est_SRP_PHAT','Est_W_SRP_PHAT','Est_GCCWLS'};
% Display names for legend
my_legend   = {'SRP-MVDR','SRP-PHAT','W-SRP-PHAT','GCC-WLS'};

% ===== Color and Style Configuration =====
% Color scheme: deep blue, deep red, deep green, deep purple
colors = [0.00 0.30 0.60;   % Deep blue
          0.80 0.20 0.20;   % Deep red
          0.10 0.60 0.40;   % Deep green
          0.50 0.20 0.70];  % Deep purple

% Consistent line styles: dash-dot + marker shapes for differentiation
line_style = ['o-.';   % SRP-MVDR   (circle + dash-dot)
              's-.';   % SRP-PHAT   (square + dash-dot)
              '^-.';   % W-SRP-PHAT (triangle + dash-dot)
              'd-.'];  % GCC-WLS    (diamond + dash-dot)

% Set default font properties for all plots
set(groot, 'defaultAxesFontName', 'Times New Roman', 'defaultTextFontName', 'Times New Roman', ...
    'defaultAxesFontSize', 12, 'defaultTextFontSize', 12);

% ===== Load and Extract Results Data =====
% Load MAT file and extract table (handles multiple naming conventions)
S = load(mat_file_all);
if isfield(S,'T1'), T = S.T1;
elseif isfield(S,'T2'), T = S.T2;
else
    fn = fieldnames(S);
    for k = 1:numel(fn)
        if istable(S.(fn{k})), T = S.(fn{k}); break; end
    end
end
if ~exist('T','var'), error('No table found.'); end

% Extract true angle column from results table
ang = T.TrueAngle;

% ===== Figure Setup =====
% Create a publication-quality figure with custom dimensions
% Size optimized for three vertical subplots

% 1) Create and configure figure for paper submission
fig = figure('Position',[0 0 560 850], 'Color',[1 1 1]);
set(fig, 'DefaultTextInterpreter', 'latex');  % Enable LaTeX formatting for math

% 2) Create tiled layout for three subplots with compact spacing
tl = tiledlayout(fig, 3, 1, 'TileSpacing','compact', 'Padding','tight');

%% ===== PLOT 1: Root Mean Square Error (RMSE) vs. Angle =====
ax1 = nexttile(tl, 1); hold(ax1, 'on');

% Compute and plot RMSE for each method
for i = 1:numel(method_cols)
    % Calculate estimation error: estimated angle - true angle
    err = T.(method_cols{i}) - ang;

    % Group errors by unique angle for per-angle statistics
    [uAng, ~, ic] = unique(ang);

    % Compute RMSE for each angle (ignoring NaN values)
    rmse = arrayfun(@(j) sqrt(mean(err(ic==j).^2, 'omitnan')), 1:numel(uAng));

    % Plot RMSE curve with method-specific color and style
    h = plot(ax1, uAng, rmse, line_style(i,:), 'Color', colors(i,:), ...
        'LineWidth', 1.5, 'MarkerSize', 6);
    % Use unfilled markers for clarity
    h.MarkerFaceColor = 'w';
end

% Axis labels and formatting
xlabel(ax1, '$\theta$ ($^\circ$)');  % Angle in degrees
ylabel(ax1, 'RMSE ($^\circ$)');

% Add subplot label
text(ax1, 0.94, 0.11, '(a)', 'Units','normalized', ...
    'HorizontalAlignment','left', 'VerticalAlignment','top', ...
    'BackgroundColor','w', 'Margin',0.001, 'Clipping','on');

% Grid and finalize axes
grid(ax1, 'on'); box(ax1, 'on');
ax1.LineWidth = 1.0;

% Legend (shared across all methods)
lgd1 = legend(ax1, my_legend, 'Location','northoutside', ...
    'Orientation','horizontal', 'NumColumns',4, 'FontSize',10);

%% ===== PLOT 2: Soft-Accuracy (S-ACC) vs. Angle =====
% S-ACC uses Cauchy soft-thresholding: score = 1 / (1 + (error/threshold)^2)
% Provides continuous accuracy metric instead of hard threshold
ax2 = nexttile(tl, 2); hold(ax2, 'on');

for i = 1:numel(method_cols)
    % Calculate estimation error
    err = T.(method_cols{i}) - ang;

    % Group by unique angles
    [uAng, ~, ic] = unique(ang);

    % Compute Cauchy soft-accuracy (S-ACC at 5 degree threshold)
    acc = NaN(size(uAng));
    for j = 1:numel(uAng)
        thr = 5.0;  % Soft-accuracy threshold (degrees)

        % Cauchy soft-thresholding formula for continuous accuracy metric
        % Properties: error=0° -> score=1; error=5° -> score=0.5
        soft_score = 1 ./ (1 + (err(ic==j) / thr).^2);

        % Average soft-accuracy across all test cases at this angle
        acc(j) = mean(soft_score, 'omitnan');
    end

    % Plot soft-accuracy curve
    h = plot(ax2, uAng, acc, line_style(i,:), 'Color', colors(i,:), ...
        'LineWidth', 1.5, 'MarkerSize', 6);
    h.MarkerFaceColor = 'w';
end

% Axis labels and configuration
xlabel(ax2, '$\theta$ ($^\circ$)');
ylabel(ax2, 'S-ACC@$5^\\circ$');  % Soft-accuracy at 5 degree threshold
ylim(ax2, [0 1]);  % Accuracy range: [0, 1]

% Add subplot label
text(ax2, 0.94, 0.11, '(b)', 'Units','normalized', ...
    'HorizontalAlignment','left', 'VerticalAlignment','top', ...
    'BackgroundColor','w', 'Margin',0.001, 'Clipping','on');

% Grid and finalize
grid(ax2, 'on'); box(ax2, 'on');
ax2.LineWidth = 1.0;

lgd2 = legend(ax2, my_legend, 'Location','northoutside', ...
    'Orientation','horizontal', 'NumColumns',4, 'FontSize',10);

%% ---------------- Tile 3: Boxplot with filled colors ----------------
ax3 = nexttile(tl, 3); hold(ax3, 'on');

binCenters = 20:20:160;
nBins = numel(binCenters);
nMethods = numel(method_cols);
boxWidth = 0.15;
allErrForY = [];

dummyHandles = gobjects(1, nMethods);

for i = 1:nMethods
    err = T.(method_cols{i}) - ang;
    allErrForY = [allErrForY; err(~isnan(err))];

    angleBins = round((ang - binCenters(1)) / 20) * 20 + binCenters(1);
    angleBins = max(min(angleBins, binCenters(end)), binCenters(1));

    for j = 1:nBins
        idx = angleBins == binCenters(j);
        if any(idx)
            pos = j + (i - (nMethods+1)/2) * boxWidth;

            % boxplot ignores NaN by default, so no extra handling is needed
            h = boxplot(ax3, err(idx), 'positions', pos, 'colors', 'k', ...
                'widths', boxWidth*0.9, 'symbol', '', 'labels', {''});
            set(h, 'LineWidth', 0.8);

            hold(ax3, 'on');

            hPatch = findobj(h, 'Tag', 'Box');
            if ~isempty(hPatch)
                patch(get(hPatch, 'XData'), get(hPatch, 'YData'), colors(i,:), ...
                    'FaceAlpha', 0.85, 'EdgeColor', 'k', 'LineWidth', 0.8);
            end
        end
    end

    dummyHandles(i) = patch(ax3, NaN, NaN, colors(i,:), 'EdgeColor', 'k', 'LineWidth', 0.8);
end

set(ax3, 'XTick', 1:nBins, 'XTickLabel', binCenters);
xlabel(ax3, '$\theta$ ($^\circ$)');
ylabel(ax3, 'Estimation Error ($^\circ$)');
set(ax3, 'XLimMode', 'auto');

% Robustly auto-adjust Y-axis limits based on the main error distribution,
% avoiding excessive stretching due to a few outliers
if ~isempty(allErrForY)
    vals = sort(allErrForY);
    nVals = numel(vals);

    iLow  = max(1, round(0.02 * nVals));
    iHigh = min(nVals, round(0.98 * nVals));

    yLow  = vals(iLow);
    yHigh = vals(iHigh);

    if yLow == yHigh
        yLow = yLow - 1;
        yHigh = yHigh + 1;
    end

    yPad = 0.12 * (yHigh - yLow);
    yLimAuto = [yLow - yPad, yHigh + yPad];

    % Center the plot around 0° error for easier comparison
    yAbs = max(abs(yLimAuto));
    ylim(ax3, [-yAbs, yAbs]);
else
    set(ax3, 'YLimMode', 'auto');
end

text(ax3, 0.94, 0.11, '(c)', 'Units','normalized', ...
    'HorizontalAlignment','left', 'VerticalAlignment','top', ...
    'BackgroundColor','w', 'Margin',0.001, 'Clipping','on');

grid(ax3, 'on'); box(ax3, 'on');
ax3.LineWidth = 1.0;

lgd3 = legend(ax3, dummyHandles, my_legend, 'Location','northoutside', ...
    'Orientation','horizontal', 'NumColumns',4, 'FontSize',10);
lgd3.ItemTokenSize = [15, 10];

% ---- Make legend text + tick labels slightly bolder ----
set([ax1, ax2, ax3], 'FontWeight', 'bold');      % Bold axis tick labels
set([lgd1, lgd2, lgd3], 'FontWeight', 'bold');   % Bold legend text

% ---- Extend the third legend length naturally to match the first two (avoid large blank space) ----
drawnow;

lgd1.Units = 'normalized';
lgd3.Units = 'normalized';
wTarget = lgd1.Position(3);

tok = lgd3.ItemTokenSize;
for it = 1:60
    drawnow;
    if lgd3.Position(3) >= 0.995 * wTarget
        break;
    end
    tok(1) = tok(1) + 5;
    lgd3.ItemTokenSize = tok;
end
