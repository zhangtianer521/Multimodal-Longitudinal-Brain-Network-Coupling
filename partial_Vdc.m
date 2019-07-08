function Pr=partial_Vdc(vd, vdc, K, nuser, lb3, lb4)

Pr=zeros(size(vd(K,:,1)));
for i=1:nuser
    Pr=Pr+2*lb3*(vdc(K,:)-vd(K,:,i));
end
Pr=Pr+2*lb4*vdc(K,:);