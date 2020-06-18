clear

loadValue = 0.002;
Sxx = get_sigma_2D(loadValue, [1, 0, 0]);
Syy = get_sigma_2D(loadValue, [0, 1, 0]);
Sxy = get_sigma_2D(loadValue, [0, 0, 1]);

C1111 = Sxx(1) / loadValue
C1122 = Sxx(2) / loadValue
C1112 = Sxx(3) / loadValue

C2222 = Syy(2) / loadValue
C1222 = Syy(3) / loadValue

C1212 = Sxy(3) / loadValue