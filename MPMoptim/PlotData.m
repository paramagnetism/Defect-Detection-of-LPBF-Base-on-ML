% Load the data from the CSV file
clc
clear all
close all

% % Open the figure file
% fig_handle = openfig('nodownsample.fig');
% % Modify the figure (optional)
% set(fig_handle, 'Color', 'w'); % Set the figure background color to white

X1data = csvread('000_030_00_Int_y.csv');
Y1data = csvread('000_030_00_Int_x.csv');
Z1data = csvread('000_030_00_Int_on_axis.csv');
F1data = csvread('000_030_00_Int_off_axis.csv');
P1data = csvread('000_030_00_Int_p.csv');

% 
% X2data = csvread('003_960_00_Int_subplot3_line1.csv');
% Y2data = csvread('003_960_00_Int_subplot3_line2.csv');
% Z2data = csvread('003_960_00_Int_subplot5_line1.csv');
% F2data = csvread('003_960_00_Int_subplot4_line1.csv');
% P2data = csvread('003_960_00_Int_subplot2_line1.csv');

c1 = X1data(:,1);
% c2 = X2data(:,1);

% Extract the x, y, and z data from the loaded data
x1 = X1data(:,2);
y1 = Y1data(:,2);
z1 = Z1data(:,2);
f1 = F1data(:,2);
p1 = P1data(:,2);

rangy = [104, 112;
         121, 129;
         138, 146];
x_1 = 76:8:116;
x_2 = 84:18:174;

rangx = [x_1;x_2];
rangx = rangx';

Rangxy = cell(18,1);
Index_R = [1:6;
           7:12;
           13:18];
Index_R = Index_R';

for i = 1:length(c1)
    % Find the appropriate group for x value
    flag_x = 0;
    flag_y = 0;
    for ix = 1:size(rangx,1)
        if x1(i) >= rangx(ix,1) && x1(i) <= rangx(ix,2)
            flag_x = 1;
            break;
        end
    end
    % Find the appropriate group for y value
    for iy = 1:size(rangy,1)
        if y1(i) >= rangy(iy,1) && y1(i) <= rangy(iy,2)
            flag_y = 1;
            break;
        end
    end
    if flag_x*flag_y == 1
        Rangxy{Index_R(ix,iy)} = [Rangxy{Index_R(ix,iy)} i];
    end
end

% Save the cell values to a CSV file
csvFileName = 'number range for cubes.csv';
writecell(Rangxy, csvFileName);  % For cell array of strings

for ih = 1:18
    data = [];
    rangh = [];
    rangh = Rangxy{ih,1};
    data = [x1(rangh) y1(rangh) z1(rangh) f1(rangh)];
    filename = sprintf('000_030_00_Int_data%d.csv', ih);
        % Write the data to a CSV file
        csvwrite(filename, data);
end
% x2 = X2data(:,2);
% y2 = Y2data(:,2);
% z2 = Z2data(:,2);
% f2 = Z2data(:,2);
% p2 = P2data(:,2);
% 
% figure 
% plot(c0, x, 'b')
% hold on
% plot(c0, y, 'g')
% title('Coordinate');
% 
% figure
% plot(c0, z, 'b') 
% title('axis-on');
% 
% figure
% plot(c0, p, 'b')
% title('laser power');

return

% for i=1346872:3.4e6
%     if p(i)>= 5
%         i
%         break
%     end     
% end
% 
% return
% rang2 = 1:length(x);
% rang = 1:3679279;
% rang = 1:3579279;
% rang1 = 3679280:7200944;
% rang = 1:3521522;
rang = 1:1459337;
rang1 = 1459338:1571807;
rang2 = 1459338:3371709;
% rang2 = 1346872:3371709;
% rang2 = 1:1571807;
% rang = 3521523:3679279;
% rang = 3521523:length(x);
% rang = 3679280:length(x);
% rang = 1:1e5;

figure
plot(c0(rang), x(rang), 'g'); 
hold on
plot(c0(rang), y(rang), 'b'); 
title('x1 and y1');

% Set the y-axis limits to 0 to 400
% ylim([0, 400]);
% % Set the x-axis limits to 0 to the maximum value in the x-axis data
% xlim([0, length(x(rang))]);

figure 
plot(c0(rang), z(rang), 'b')
% Set the y-axis limits to 0 to 400
% ylim([0, 4e4]);
% % Set the x-axis limits to 0 to the maximum value in the x-axis data
% xlim([0, length(x(rang))]);
title('z1');

figure 
plot(c0(rang), p(rang), 'b')
title('p1');

% Draw heatmap of Z based on X and Y
figure(1);
rang = 1:length(c1);
scatter3(x1(rang), y1(rang), z1(rang), 10, z1(rang), 'filled');
% Change the view direction to focus on the z-axis
view(0, 90);
xlabel('x');
ylabel('y');
axis equal
title('Heatmap of On-Axis Z 000-030-00');
colormap jet;
colorbar;

figure(2);
rang = 1:length(c2);
scatter3(y2(rang), x2(rang), z2(rang), 10, z2(rang), 'filled');
% Change the view direction to focus on the z-axis
view(0, 90);
xlabel('x');
ylabel('y');
axis equal
title('Heatmap of On-Axis Z 003-960-00');
colormap jet;
colorbar;
% caxis([0, 6000]);

%%
figure
plot(c0(rang1), x(rang1), 'g'); 
hold on
plot(c0(rang1), y(rang1), 'b'); 
title('x2 and y2');
% Set the y-axis limits to 0 to 400
% ylim([0, 400]);
% % Set the x-axis limits to 0 to the maximum value in the x-axis data
% xlim([0, length(x(rang1))]);

figure 
plot(c0(rang1), z(rang1), 'b')
% Set the y-axis limits to 0 to 400
% ylim([0, 4e4]);
% % Set the x-axis limits to 0 to the maximum value in the x-axis data
% xlim([0, length(x(rang1))]);
title('z2');

figure 
plot(c0(rang1), p(rang1), 'b')
title('p2');

% Draw heatmap of Z based on X and Y
figure;

scatter3(x(rang1), y(rang1), z(rang1), 20, z(rang1), 'filled');
% Change the view direction to focus on the z-axis
view(0, 90);
xlabel('x');
ylabel('y');
axis equal
title('Heatmap of On-Axis 2');
colormap jet;
colorbar;
caxis([0, 2000]);

%%
x0 = [1, 1350401, 1929294, 2079192, length(x)];
for i=1.552e6:x0(3)
    if p(i)<1.2
        i
        break
    end     
end


figure
plot(c0(rang2), x(rang2), 'g'); 
hold on
plot(c0(rang2), y(rang2), 'b'); 
title('x3 and y3');
% Set the y-axis limits to 0 to 400
% ylim([0, 400]);
% % Set the x-axis limits to 0 to the maximum value in the x-axis data
% xlim([0, length(x(rang1))]);

figure 
plot(c0(rang2), z(rang2), 'b')
% Set the y-axis limits to 0 to 400
% ylim([0, 4e4]);
% % Set the x-axis limits to 0 to the maximum value in the x-axis data
% xlim([0, length(x(rang1))]);
title('z3');

figure 
plot(c0(rang2), p(rang2), 'b')
title('p3');

% Draw heatmap of Z based on X and Y
Z2data = csvread('subplot4_line1.csv');
z2 = Z2data(:,2);
figure;
rang2 = 1736684:1832415; 
scatter3(x(rang2), y(rang2), z2(rang2), 20, z2(rang2), 'filled');
% Change the view direction to focus on the z-axis
view(0, 90);
xlabel('x');
ylabel('y');
axis equal
title('Heatmap of Off-Axis 3');
colormap jet;
colorbar;
caxis([500, 900]);

%%
x0 = [1, 1350401, 1929294, 2079192, length(x)];
for ii = 1:4
rang = x0(ii):x0(ii+1);
figure(ii)
scatter3(x(rang), y(rang), z(rang), 20, z(rang), 'filled');
% Change the view direction to focus on the z-axis
view(0, 90);
xlabel('x');
ylabel('y');
axis equal
title('Heatmap of On-Axis');
colormap jet;
colorbar;
caxis([0, 2000]);
end

x1 = [1350401, 1446889, 1543465, 1639569, 1735928, 1832415 ,1929294];
for ii = 1:length(x1)-1
rang = x1(ii):x1(ii+1);
figure(ii)
scatter3(x(rang), y(rang), z(rang), 20, z(rang), 'filled');
% Change the view direction to focus on the z-axis
view(0, 90);
xlabel('x');
ylabel('y');
axis equal
title('Heatmap of On-Axis for scanning');
colormap jet;
colorbar;
end
