%% 航拍建筑物提取 —— MATLAB 结果分析与可视化
% 读取 PyCharm 演示脚本导出的图块与指标，以及三组模型汇总结果，
% 生成论文汇报用图。运行方式（项目根目录）：
%   matlab -batch "run('matlab/analyze_results.m')"
%
% 输出：
%   matlab_output/matlab_paper_results.png
%   matlab_output/matlab_demo_metrics.png
%   matlab_output/matlab_demo_montage.png
%   matlab_output/matlab_summary.txt

clear; clc; close all;
root = fileparts(fileparts(mfilename('fullpath')));
exDir = fullfile(root, 'matlab_exchange');
outDir = fullfile(root, 'matlab_output');
if ~exist(outDir, 'dir'); mkdir(outDir); end

set(0, 'DefaultAxesFontName', 'Microsoft YaHei');
set(0, 'DefaultTextFontName', 'Microsoft YaHei');
set(0, 'DefaultAxesFontSize', 10);
set(0, 'DefaultTextFontSize', 10);

fprintf('MATLAB %s\n', version);
fprintf('项目目录: %s\n', root);

%% 1. 读取数据
demoCsv = fullfile(exDir, 'demo_manifest.csv');
paperCsv = fullfile(root, 'results', 'model_summary.csv');
assert(exist(demoCsv, 'file') == 2, '缺少 demo_manifest.csv，请先运行 PyCharm 演示脚本');
assert(exist(paperCsv, 'file') == 2, '缺少 model_summary.csv');

T = readtable(demoCsv, 'Delimiter', ',', 'TextType', 'string');
P = readtable(paperCsv, 'Delimiter', ',', 'TextType', 'string');
S = jsondecode(fileread(fullfile(exDir, 'demo_summary.json')));

n = height(T);
group = strings(n, 1);
sceneLabel = strings(n, 1);
for i = 1:n
    nm = char(T.name(i));
    key = extractBefore(string(T.name(i)), "_");
    switch key
        case "shadow";  sceneLabel(i) = "弱光照/阴影";
        case "dense";   sceneLabel(i) = "密集建筑";
        case "small";   sceneLabel(i) = "小目标";
        case "clutter"; sceneLabel(i) = "复杂背景";
        case "success"; sceneLabel(i) = "成功案例";
        otherwise;      sceneLabel(i) = "失败案例";
    end
    if startsWith(string(T.name(i)), "success")
        group(i) = "成功案例";
    elseif startsWith(string(T.name(i)), "failure")
        group(i) = "失败案例";
    else
        group(i) = "典型场景";
    end
end
groupColor = zeros(n, 3);
groupColor(group == "典型场景", :) = repmat([0.23 0.49 0.63], sum(group == "典型场景"), 1);
groupColor(group == "成功案例", :) = repmat([0.30 0.62 0.38], sum(group == "成功案例"), 1);
groupColor(group == "失败案例", :) = repmat([0.78 0.33 0.24], sum(group == "失败案例"), 1);

fprintf('演示图块: %d 个，平均 IoU %.4f，非失败案例平均 IoU %.4f\n', ...
    n, S.mean_iou, S.non_failure_mean_iou);

%% 2. 三组模型总体指标
M = [P.iou, P.f1, P.biou, P.small_recall];
fig1 = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 1280 560]);
b = bar(M, 'grouped');
b(1).FaceColor = [0.50 0.55 0.55];
b(2).FaceColor = [0.23 0.49 0.63];
b(3).FaceColor = [0.78 0.33 0.24];
b(4).FaceColor = [0.85 0.66 0.25];
set(gca, 'XTickLabel', {'U-Net 基线', '+ASPP 多尺度', '+ASPP+边界监督'});
ylabel('得分'); ylim([0 0.85]); grid on; box off;
legend({'IoU', 'F1', '边界 IoU', '小目标召回'}, 'Location', 'northoutside', ...
    'Orientation', 'horizontal', 'Box', 'off');
title('三组模型总体指标（Massachusetts 测试集，阈值 0.5）', 'FontSize', 13);
for i = 1:numel(b)
    x = b(i).XEndPoints; y = b(i).YEndPoints;
    text(x, y + 0.012, compose('%.3f', y), 'HorizontalAlignment', 'center', ...
        'FontSize', 8, 'Color', [0.2 0.2 0.2]);
end
print(fig1, '-dpng', '-r220', fullfile(outDir, 'matlab_paper_results.png'));
savefig(fig1, fullfile(outDir, 'matlab_paper_results.fig'));
close(fig1);

%% 3. 演示图块逐图指标
fig2 = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 1280 540]);
subplot(1, 2, 1);
bb = bar(T.iou, 0.62, 'FaceColor', 'flat');
bb.CData = groupColor;
hold on;
yline(S.non_failure_mean_iou, '--', '非失败案例均值', 'Color', [0.2 0.2 0.2], ...
    'LineWidth', 1.1, 'LabelHorizontalAlignment', 'left');
hold off;
set(gca, 'XTick', 1:n, 'XTickLabel', cellstr(sceneLabel), 'XTickLabelRotation', 35);
ylabel('IoU'); ylim([0 0.9]); grid on; box off;
title('演示图块 IoU（8 个代表性图块）', 'FontSize', 12);

subplot(1, 2, 2);
hold on;
h1 = scatter(T.gt_count(group == "典型场景"), T.iou(group == "典型场景"), 90, ...
    groupColor(group == "典型场景", :), 'filled', 'MarkerEdgeColor', 'w');
h2 = scatter(T.gt_count(group == "成功案例"), T.iou(group == "成功案例"), 90, ...
    groupColor(group == "成功案例", :), 'filled', 'MarkerEdgeColor', 'w');
h3 = scatter(T.gt_count(group == "失败案例"), T.iou(group == "失败案例"), 90, ...
    groupColor(group == "失败案例", :), 'filled', 'MarkerEdgeColor', 'w');
yline(S.non_failure_mean_iou, '--', 'Color', [0.4 0.4 0.4]);
hold off;
xlabel('真值建筑数量（栋）'); ylabel('IoU');
xlim([0 max(T.gt_count) + 15]); ylim([0 0.9]); grid on; box off;
legend([h1 h2 h3], {'典型场景', '成功案例', '失败案例'}, 'Location', 'southwest', 'Box', 'off');
title('建筑密度与提取精度', 'FontSize', 12);
print(fig2, '-dpng', '-r220', fullfile(outDir, 'matlab_demo_metrics.png'));
savefig(fig2, fullfile(outDir, 'matlab_demo_metrics.fig'));
close(fig2);

%% 4. 四类典型场景的可视化拼图
keys = ["shadow", "dense", "small", "clutter"];
labels = ["弱光照 / 阴影", "密集建筑", "小目标", "复杂背景"];
fig3 = figure('Visible', 'off', 'Color', 'w', 'Position', [40 40 1400 1080]);
t = tiledlayout(4, 4, 'Padding', 'compact', 'TileSpacing', 'compact');
for r = 1:numel(keys)
    idx = find(startsWith(string(T.name), keys(r) + "_"), 1);
    if isempty(idx); continue; end
    nm = char(T.name(idx));
    img = imread(fullfile(exDir, 'patches', [nm '_image.png']));
    gt = imread(fullfile(exDir, 'patches', [nm '_gt.png'])) > 127;
    pred = imread(fullfile(exDir, 'patches', [nm '_pred.png'])) > 127;
    ov = overlayMask(img, pred, [0.10 0.45 0.85]);
    er = errorImage(pred, gt);

    nexttile; imshow(img); title(sprintf('%s 原图', labels(r)), 'FontSize', 9);
    nexttile; imshow(overlayMask(img, gt, [0.15 0.75 0.35])); title('真值（绿色）', 'FontSize', 9);
    nexttile; imshow(ov); title(sprintf('预测 IoU=%.3f', T.iou(idx)), 'FontSize', 9);
    nexttile; imshow(er); title('误差：绿TP 红FP 蓝FN', 'FontSize', 8);
end
title(t, '完整模型在四类典型场景上的提取效果', 'FontSize', 14, 'FontName', 'Microsoft YaHei');
print(fig3, '-dpng', '-r200', fullfile(outDir, 'matlab_demo_montage.png'));
savefig(fig3, fullfile(outDir, 'matlab_demo_montage.fig'));
close(fig3);

%% 5. 文本摘要
fid = fopen(fullfile(outDir, 'matlab_summary.txt'), 'w', 'n', 'UTF-8');
fprintf(fid, '航拍建筑物提取 —— MATLAB 分析摘要\n');
fprintf(fid, '生成时间: %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, '一、PyCharm 演示结果\n');
fprintf(fid, '  图块数: %d\n', n);
fprintf(fid, '  平均 IoU: %.4f\n', S.mean_iou);
fprintf(fid, '  非失败案例平均 IoU: %.4f\n', S.non_failure_mean_iou);
fprintf(fid, '  失败案例平均 IoU: %.4f\n', S.failure_mean_iou);
fprintf(fid, '  平均边界 IoU: %.4f\n', S.mean_biou);
fprintf(fid, '  平均小目标召回: %.4f\n', S.mean_small_recall);
fprintf(fid, '  平均粘连率: %.4f\n\n', S.mean_merge_rate);
fprintf(fid, '二、论文总体结果（测试集，阈值 0.5）\n');
for i = 1:height(P)
    fprintf(fid, '  %-6s IoU=%.4f F1=%.4f 边界IoU=%.4f 小目标召回=%.4f 粘连率=%.4f\n', ...
        char(P.model(i)), P.iou(i), P.f1(i), P.biou(i), P.small_recall(i), P.merge_rate(i));
end
fprintf(fid, '\n三、主要结论\n');
fprintf(fid, '  1. 基线 U-Net 与完整模型的总体 IoU 并列最高（0.5830），但边界与实例可分性不足。\n');
fprintf(fid, '  2. ASPP 将小目标召回提升 1.3%%，粘连率降低 10.8%%，但总体 IoU 下降 0.77%%。\n');
fprintf(fid, '  3. 边界监督把边界 IoU 提升到 0.3703，但总体 IoU 与基线持平。\n');
fprintf(fid, '  4. 下一阶段应使用 0.3 m 高分辨率数据、预训练编码器与实例分离后处理。\n');
fclose(fid);

fprintf('MATLAB_ANALYSIS_OK\n');
fprintf('输出目录: %s\n', outDir);

%% 局部函数
function out = overlayMask(img, mask, color)
    out = double(img) / 255;
    m = logical(mask);
    for c = 1:3
        ch = out(:, :, c);
        ch(m) = 0.58 * ch(m) + 0.42 * color(c);
        out(:, :, c) = ch;
    end
    out = uint8(max(0, min(1, out)) * 255);
end

function out = errorImage(pred, gt)
    out = zeros([size(gt), 3], 'uint8');
    colors = uint8([64 140 77; 217 51 41; 38 97 204]);
    tp = pred & gt; fp = pred & ~gt; fn = ~pred & gt;
    for c = 1:3
        ch = out(:, :, c);
        ch(tp) = colors(1, c);
        ch(fp) = colors(2, c);
        ch(fn) = colors(3, c);
        out(:, :, c) = ch;
    end
end
