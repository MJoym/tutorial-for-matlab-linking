function [corners, comb] = corners_combinations(stim_ind)
% Create all possible combinations for collinearity check (between corners):
% added more changes
% Find max and min (corners):
rowMax = find(stim_ind(:,1)==max(stim_ind(:,1)));
rowMin = find(stim_ind(:,1)==min(stim_ind(:,1)));
bottomRight = [max(stim_ind(:,1)), max(stim_ind(rowMax,2))];
bottomLeft = [max(stim_ind(:,1)), min(stim_ind(rowMax,2))];
topRight = [min(stim_ind(:,1)), max(stim_ind(rowMin,2))];
topLeft = [min(stim_ind(:,1)), min(stim_ind(rowMin,2))];

if bottomLeft == bottomRight
    if topLeft == topRight
    corners = [bottomLeft; topLeft];
    else
        corners = [bottomLeft; topLeft; topRight];
    end
else
    if topLeft == topRight
        corners = [bottomLeft; topLeft; bottomRight];
    else
        corners = [bottomLeft; topLeft; topRight; bottomRight];
    end
end

comb = nchoosek(1:size(corners,1),2);
end