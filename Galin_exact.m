% this function returnes stress and displacement elastoplastic distribution
% around hole in the infinite plane, subjected to
% (sx+sy)/2 = -P_inf   at infinity
% (sy-sx)/2 = tau_inf  at infinity
% sr        = -P_in    at the hole
% Ys        - yield strength
% R0        - radius of the hole
% x,y       - 2D numerical grid
% abs(tau) < 0.4142  Limitations to parameters
function [sxx,syy,sxy,x_B,y_B] = Galin_exact(Ys,P_inf,P_in,tau_inf,R0,x,y)
DeltaP   = -(P_inf - P_in);       % Effective stress
xi       = sign(DeltaP);% Different formula for compression and tension
kappa    = tau_inf/Ys*xi;   % Parameters defining legitimation of the solution
c0       = R0*exp(abs(P_inf - P_in)/2/Ys - 1/2);
ra       = c0*(1+kappa);
rb       = c0*(1-kappa);
srr      = 0*x - P_in;
stt      = 0*x - P_in;
srt      = 0*x;
% PLASTIC ZONE
% Polar and cartesian coordinates in plane Z
z        = complex(x,y);
r        = abs(z);
th       = angle(z);
% Boundary of elastic-plastic zones
th_B     = 0:0.01:2*pi;
z_B      = c0*(exp(1i*th_B) + kappa./exp(1i*th_B));
x_B      = real(z_B);
y_B      = imag(z_B);
% ELASTIC ZONE
i_e      = find(x.^2/ra^2+y.^2/rb^2> 1);
singx    = sign(x(i_e)) + sign(y(i_e)).*(x(i_e)==0);
Zeta     =(z(i_e) + singx.*sqrt(z(i_e).^2 - 4*c0^2*kappa))/2/c0;
% Intermediate elastic functions in plane Zeta
w        = c0*(Zeta + kappa./Zeta);
wv       = c0*(1./Zeta + kappa*Zeta);                  % w(Zeta) with ~
dw       = c0*(1 - kappa./Zeta.^2);                    % w'(Zeta)
phi      = (-P_in + Ys*xi)/2 + Ys*xi*log(w./Zeta/R0);
psi      = Ys*xi./Zeta.*wv./dw;
dphi     = - 2*xi*Ys*kappa ./Zeta ./( Zeta.^2 + kappa );  % phi'(Zeta)
F        = 2*( w./dw.*dphi + w./conj(w).*psi );
srr(i_e) = 2*real(phi) - real(F)/2;
stt(i_e) = 2*real(phi) + real(F)/2;
srt(i_e) = imag(F)/2;
% Stresses in the plastic zone
i_p      = find( (x.^2/ra^2+y.^2/rb^2<=1) & r>R0 );
srr(i_p) = -P_in + 2*Ys*xi*log( r(i_p)/R0 );
stt(i_p) = srr(i_p) + 2*xi*Ys;
srt(i_p) = 0;
% rotation to Cartesian
Pr       = -(srr + stt)/2;
dS       =  (srr - stt)/2;
sxx      = -Pr + dS.*cos(2*th) - srt.*sin(2*th);
syy      = -Pr - dS.*cos(2*th) + srt.*sin(2*th);
sxy      =       dS.*sin(2*th) + srt.*cos(2*th);
