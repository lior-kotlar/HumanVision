function show(im1,sc)

hold off;
if nargin<2
    imagesc(im1);
else
    imagesc(im1,sc);
end
colormap(gray);
axis image;