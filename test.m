load('iris.txt');
X=iris(:,1:4)';
t=iris(:,5)';
RNr=newrb(X,t);
y=sim(RNr,X)
keep


aciertos=0;
for i=1:m
    if (round(y(i))==t(i))
        aciertos=aciertos+1;
    end
end
porcentaje=(aciertos/m)*100
save dayanna1 M1E porcentaje;
aciertos=0;
for i=1:m
    if (round(y(i))==t(i))
        aciertos=aciertos+1;
    end
end
porcentaje=(aciertos/m)*100
save dayanna1 RNr porcentaje;