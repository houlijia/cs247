function y = f_handleA(A,u,trnsp)

if trnsp
    y = (u'*A)';
else
    y = A*u;
end

end