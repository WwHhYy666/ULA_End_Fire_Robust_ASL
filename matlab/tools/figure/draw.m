clear; close all; clc;

mat_file_all = 'C:\Users\25401\Desktop\End-Fire\result\DOA_results_2m.mat';
method_cols = {'Est_MVDR','Est_SRP_PHAT','Est_W_SRP_PHAT','Est_GCCWLS'};
my_legend   = {'SRP-MVDR','SRP-PHAT','W-SRP-PHAT','GCC-WLS'};

% 配色方案：深蓝、深红、深绿、深紫
colors = [0.00 0.30 0.60;
          0.80 0.20 0.20;
          0.10 0.60 0.40;
          0.50 0.20 0.70];

% 图中一致的线型：点划线 + 对应 marker
line_style = ['o-.';   % SRP-MVDR   (circle + dash-dot)
              's-.';   % SRP-PHAT   (square + dash-dot)
              '^-.';   % W-SRP-PHAT (triangle + dash-dot)
              'd-.'];  % GCC-WLS    (diamond + dash-dot)

set(groot, 'defaultAxesFontName', 'Times New Roman', 'defaultTextFontName', 'Times New Roman', ...
    'defaultAxesFontSize', 12, 'defaultTextFontSize', 12);

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

ang = T.TrueAngle;

% 1) figure 尺寸按论文图比例收紧
fig = figure('Position',[0 0 560 850], 'Color',[1 1 1]);
set(fig, 'DefaultTextInterpreter', 'latex');

% 2) 用 tiledlayout 保证三张子图长宽一致 + 子图间距一致
tl = tiledlayout(fig, 3, 1, 'TileSpacing','compact', 'Padding','tight');

%% ---------------- Tile 1: RMSE ----------------
ax1 = nexttile(tl, 1); hold(ax1, 'on');

for i = 1:numel(method_cols)
    err = T.(method_cols{i}) - ang;

    [uAng, ~, ic] = unique(ang);

    % RMSE 计算忽略 NaN
    rmse = arrayfun(@(j) sqrt(mean(err(ic==j).^2, 'omitnan')), 1:numel(uAng));

    h = plot(ax1, uAng, rmse, line_style(i,:), 'Color', colors(i,:), ...
        'LineWidth', 1.5, 'MarkerSize', 6);
    h.MarkerFaceColor = 'w';
end

xlabel(ax1, '$\theta$ ($^\circ$)');
ylabel(ax1, 'RMSE ($^\circ$)');

text(ax1, 0.94, 0.11, '(a)', 'Units','normalized', ...
    'HorizontalAlignment','left', 'VerticalAlignment','top', ...
    'BackgroundColor','w', 'Margin',0.001, 'Clipping','on');

grid(ax1, 'on'); box(ax1, 'on');
ax1.LineWidth = 1.0;

lgd1 = legend(ax1, my_legend, 'Location','northoutside', ...
    'Orientation','horizontal', 'NumColumns',4, 'FontSize',10);

%% ---------------- Tile 2: Accuracy ----------------
ax2 = nexttile(tl, 2); hold(ax2, 'on');

for i = 1:numel(method_cols)
    err = T.(method_cols{i}) - ang;

    [uAng, ~, ic] = unique(ang);

    % ---- 修改点：采用 Cauchy Soft-Accuracy (S-ACC) ----
    acc = NaN(size(uAng));
    for j = 1:numel(uAng)
        thr = 5.0; % 设定的容忍度尺度参数
        
        % 柯西软阈值公式：1 / (1 + (e/th)^2)
        % 误差=0度得1分；误差=5度得0.5分，避免阶跃断崖
        soft_score = 1 ./ (1 + (err(ic==j) / thr).^2);
        
        acc(j) = mean(soft_score, 'omitnan');
    end
    % ---------------------------------------------------

    h = plot(ax2, uAng, acc, line_style(i,:), 'Color', colors(i,:), ...
        'LineWidth', 1.5, 'MarkerSize', 6);
    h.MarkerFaceColor = 'w';
end

xlabel(ax2, '$\theta$ ($^\circ$)');
% ---- 修改点：更新 Y 轴标签 ----
ylabel(ax2, 'S-ACC@$5^\circ$');
ylim(ax2, [0 1]);

text(ax2, 0.94, 0.11, '(b)', 'Units','normalized', ...
    'HorizontalAlignment','left', 'VerticalAlignment','top', ...
    'BackgroundColor','w', 'Margin',0.001, 'Clipping','on');

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

            % boxplot 会自动忽略 NaN，所以这里无需额外处理
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

% 根据误差主体分布自适应 Y 轴范围（稳健，避免少量离群值拉伸过度）
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

    % 让误差图在 0 度附近更居中、更易比较
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
set([ax1, ax2, ax3], 'FontWeight', 'bold');      % 坐标轴刻度数字加粗
set([lgd1, lgd2, lgd3], 'FontWeight', 'bold');   % 图例文字加粗



% ---- 让第三张图例长度“自然”变长到与前两个一致（不留大段空白）----
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
