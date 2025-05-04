function [Iw,warpMask] = warp(Im,v)

[row_num,col_num] = size(Im);

xx = repmat(1:col_num,row_num,1);
yy = repmat(flipud([1:row_num]'),1,col_num);

Iw = interp2(xx,yy,Im,xx+v(1),yy+v(2));
nan_ind = find(isnan(Iw));
warpMask = ones(size(Iw));
warpMask (nan_ind) = 0;
Iw(nan_ind) = 0;



