function fr=fVdc(vd, vdc, nuser, lb3, lb4)

fr=0;
for i=1:nuser
    fr=fr + lb3*norm((vd(:,:,i)-vdc),'fro').^2;
end
fr=fr+lb4*norm(vdc,'fro');
    
