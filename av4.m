function A_av = av4(A)
A_av = 0.25*(A(1:end-1,1:end-1) + A(1:end-1,2:end) ...
    +      A(2:end  ,1:end-1) + A(2:end  ,2:end));
end