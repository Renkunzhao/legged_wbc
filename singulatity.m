clc; clear; close all;

% ===== 参数范围可自由指定 =====
q2_min = -0.5236;
q2_max = 4.5379;
q2_min1 = -1.5708;
q2_max1 = 3.4907;
q3_min = -2.7227;    % 起始角度 (rad)
q3_max = -0.83776;    % 结束角度 (rad)

% ===== 网格生成 =====
[q2, q3] = meshgrid(linspace(-2*pi, 2*pi, 200), ...
                    linspace(-2*pi, 2*pi, 200));

% ===== 函数定义 =====
f = cos(q2) + cos(q2 + q3);

% ===== 绘制 3D 曲面 =====
figure('Color','w');
surf(q2, q3, f, 'EdgeColor', 'none', 'FaceAlpha', 0.95);
colormap(jet);
colorbar;
hold on;

% ===== 绘制 f=0 等值线（奇异线） =====
contour3(q2, q3, f, [0 0], 'LineWidth', 2, 'LineColor', 'r');

% ===== 绘制 z=0 平面 =====
planeZ = zeros(size(q2));
surf(q2, q3, planeZ, 'FaceColor', [0.7 0.7 0.7], ...
     'FaceAlpha', 0.3, 'EdgeColor', 'none');

% ===== 绘制 q2/q3 min/max 边界线 =====
% q2 = q2_min
plot3(q2_min*ones(1,200), linspace(q3_min,q3_max,200), ...
      zeros(1,200), 'k--', 'LineWidth', 1.5);
% q2 = q2_max
plot3(q2_max*ones(1,200), linspace(q3_min,q3_max,200), ...
      zeros(1,200), 'k--', 'LineWidth', 1.5);
% q3 = q3_min
plot3(linspace(q2_min,q2_max,200), q3_min*ones(1,200), ...
      zeros(1,200), 'k--', 'LineWidth', 1.5);
% q3 = q3_max
plot3(linspace(q2_min,q2_max,200), q3_max*ones(1,200), ...
      zeros(1,200), 'k--', 'LineWidth', 1.5);

% q2 = q2_min
plot3(q2_min1*ones(1,200), linspace(q3_min,q3_max,200), ...
      zeros(1,200), 'w--', 'LineWidth', 1.5);
% q2 = q2_max
plot3(q2_max1*ones(1,200), linspace(q3_min,q3_max,200), ...
      zeros(1,200), 'w--', 'LineWidth', 1.5);
% q3 = q3_min
plot3(linspace(q2_min1,q2_max1,200), q3_min*ones(1,200), ...
      zeros(1,200), 'w--', 'LineWidth', 1.5);
% q3 = q3_max
plot3(linspace(q2_min1,q2_max1,200), q3_max*ones(1,200), ...
      zeros(1,200), 'w--', 'LineWidth', 1.5);

xline(0, 'w-', 'LineWidth', 1.5);
yline(0, 'w-', 'LineWidth', 1.5);

% ===== 坐标与标题 =====
xlabel('q_2 (rad)');
ylabel('q_3 (rad)');
zlabel('f(q_2, q_3) = cos(q_2) + cos(q_2 + q_3)');
title('Jacobian Determinant Surface and Singularity Line');
legend('Jacobian Surface','f=0 Line','z=0 Plane','Boundary','q_2=(-q_3+\pi)/2','Location','bestoutside');

grid on; shading interp; axis tight;
view([-45, 25]);

q2_min = (- q3_min - pi)/2
q2_max = (- q3_max + pi)/2
