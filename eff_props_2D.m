clear

loadValue = 0.002;
nTimeSteps = 2;
Sxx = get_sigma_2D(loadValue, [1, 0, 0], nTimeSteps);
Syy = get_sigma_2D(loadValue, [0, 1, 0], nTimeSteps);
Sxy = get_sigma_2D(loadValue, [0, 0, 1], nTimeSteps);

C1111 = zeros(nTimeSteps, 1);
C1122 = zeros(nTimeSteps, 1);
C1112 = zeros(nTimeSteps, 1);
C2222 = zeros(nTimeSteps, 1);
C1222 = zeros(nTimeSteps, 1);
C1212 = zeros(nTimeSteps, 1);

for it = 1:nTimeSteps
  C1111(it) = Sxx(it, 1) / loadValue / it * nTimeSteps
  C1122(it) = Sxx(it, 2) / loadValue / it * nTimeSteps
  C1112(it) = Sxx(it, 3) / loadValue / it * nTimeSteps

  C2222(it) = Syy(it, 2) / loadValue / it * nTimeSteps
  C1222(it) = Syy(it, 3) / loadValue / it * nTimeSteps

  C1212(it) = Sxy(it, 3) / loadValue / it * nTimeSteps
endfor