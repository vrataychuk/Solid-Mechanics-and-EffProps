clear

loadValue = -0.002;
nTimeSteps = 1;

Sxx = get_sigma_2D(loadValue, [1.0, 1.0, 0], nTimeSteps) / loadValue;