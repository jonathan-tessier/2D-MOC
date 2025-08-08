# 2D-MOC
A Zonally Averaged Model of the Meridional Overturning Circulation

We evolve the zonally averaged buoyancy in time according to 
$$
    \partial_t \overline{b}  - \frac{1}{L_x}\partial_z\Psi\frac{1}{a}\partial_\theta\overline{b}  + \frac{1}{L_x}\frac1a\partial_\theta\Psi\partial_z\overline{b}  &= \partial_z(\kappa_z\partial_z\overline{b} ) + \text{conv.}
$$
where the overturning streamfunction defined as
$$
    \Psi = 
    \begin{cases}
        \overline{\Psi}_{SO} + \Psi_{GM} &\theta<\theta_c\\
        \overline{\Psi}_{TW} + \overline\Psi_0 + \Psi_{GM} &\theta \geq \theta_c
    \end{cases}
$$
is diagnosed from $\overline{b}$ at every timestep, with flow components
$$
    \overline{\Psi}_{SO} &= -L_x\frac{\tau}{\rho_0 f}\\
    \Psi_{GM} &= -L_xK_{GM}\frac{\frac1a \partial_\theta \overline{b} }{\partial_z \overline{b} } \\
    \overline\Psi_{TW} &= \int_\theta^{\theta_n}\left(\int_{-H}^z\left( \int_{-H}^z -\frac{1}{fa}\partial_\theta \overline{b}  \,dz + C(\theta) \right)\,dz \right)\, a d\theta\\
    \overline\Psi_0  &= \frac{\sin\theta_n - \sin\theta}{\sin\theta_n - \sin\theta_c}\left(\overline{\Psi}_{SO}-\overline{\Psi}_{TW}\right)|_{\theta=\theta_c}(z).
$$
