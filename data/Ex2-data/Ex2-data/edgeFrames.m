function [im1,im2] = edgeFrames(vx,vy)
% two frames of a vertical edge moving with velocity vx, vy

[xx,yy]=meshgrid(1:128);
im1=double(xx<64);
xx=xx-vx;
yy=xx-vy;
im2=double(xx<64);