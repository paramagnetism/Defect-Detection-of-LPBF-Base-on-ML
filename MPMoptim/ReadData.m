clear all
clc
% Load the .fig file
folder = '../MPMfig/';
Files = dir(fullfile([folder,'*.fig']));
LengthFiles = length(Files);

for i = 1: LengthFiles
filename = Files(i).name;
fig_handle = openfig([folder, filename]);
filename = filename(1:length(filename)-4);
folderr = [folder, filename, '/'];
if exist(folderr)==0
    mkdir(folderr);
end
%% 
subplots = findobj(fig_handle, 'type', 'axes');

% Initialize a cell array to store the data
data = cell(length(subplots), 2);
%%
% Loop through the subplots and extract the data for each curve
for i = 1:length(subplots)
    % Get handles to all the plot objects in this subplot
%     if i == 1
%         plots = findobj(subplots(i), 'type', 'line');
%     else
        plots = findobj(subplots(i), 'type', 'line');
%     end
    xtitle = get(subplots(i), 'XLabel');
    ytitle = get(subplots(i), 'YLabel');
    
    % Loop through the plot objects and extract the data for each curve
    for j = 1:length(plots)
        % Get the x and y data for this curve
        x = get(plots(j), 'XData');
        y = get(plots(j), 'YData');
        
        % Add the data to the cell array
        data{i, j} = [x' y'];
    end
    % Print the axis titles to the console
    fprintf('Subplot %d:\n', i);
    fprintf('X axis title: %s\n', get(xtitle, 'String'));
    fprintf('Y axis title: %s\n', get(ytitle, 'String'));
end
%%
% Loop through the cell array and write the data to CSV files
for i = 3:size(data, 1)
    for j = 1: size(data, 2)+ 1 - round((i+1)/3)
        
        % Construct a filename for this curve's data
        if i == 3 && j == 1
            filename2 = [folderr,filename,'_x.csv'];
        elseif i == 3 && j == 2
            filename2 = [folderr,filename,'_y.csv'];
        elseif i == 4 && j == 1
            filename2 = [folderr,filename,'_on.csv'];
        elseif i == 5 && j == 1
            filename2 = [folderr,filename,'_off.csv'];
        end
        % Write the data to a CSV file
        csvwrite(filename2, data{i, j});
    end
end

% Close the figure file
close(fig_handle);
end