function [im]=twoSquares(shadowFlag)
% make synthetic image of two squares, one in shadow and the other not

R=ones(50);
R(30:40,30:40)=2;
R(10:20,10:20)=2;

[x,y]=meshgrid(1:50);
rr=(x-35).^2+(y-35).^2;
if (shadowFlag==1)
    L=1-0.3*softThresh(rr,13^2,0.05);
else
    L=1-0.3*softThresh(rr,3^2,0.04);
end



im=R.*L;

function [y]=softThresh(x,t,s)
% a soft threshold function useful in many places in life
% t is the threshold
% s is softness
if ~exist('s')
  s=5;
end
y=1./(1+exp(-s*(x-t)));
y=1-y;
 
 