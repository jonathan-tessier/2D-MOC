# 2D-MOC: A Zonally Averaged Model of the global ocean Meridional Overturning Circulation

We evolve the zonally averaged buoyancy ($\overline{b}$) in time according to (for latitude $\theta$, longitude $\lambda$, and local vertical $z$) 

$$
\partial_t \overline{b}  - \frac{1}{L_x} \partial_z \Psi \frac{1}{a} \partial_\theta \overline{b}  + \frac{1}{L_x}\frac{1}{a}\partial_\theta \Psi \partial_z\overline{b}  = \partial_z(\kappa_z\partial_z\overline{b} )
$$

where the overturning streamfunction, defined as

$$
\mathrm{\Psi}(\theta, z) = \begin{cases}
    \overline{\Psi} _{SO} + \Psi _{GM} & \text{in the channel, } \theta<\theta_c \\ 
    \overline{\Psi} _{TW} + \overline\Psi_0 + \Psi _{GM} & \text{in the basin, } \theta \geq \theta_c
\end{cases}
$$

is diagnosed from $\overline{b}$ at every timestep, with flow components

$$
    \overline{\Psi}_{SO} = -L_x\frac{\tau}{\rho_0 f}
$$

$$
    \Psi_{GM} = -L_x K_{GM} \frac{\frac{1}{a} \partial_\theta \overline{b} }{\partial_z \overline{b} } 
$$

$$
    \overline{\Psi}_{TW} = \int _\theta^{\theta_n} \left( \int _{-H}^z \left( \int _{-H}^z -\frac{1}{fa} \partial _\theta \overline{b} dz + C(\theta) \right) dz \right)  a d \theta
$$

$$
    \overline{\Psi}_0 = \frac{\sin\theta_n - \sin\theta}{\sin\theta_n - \sin\theta_c}\left(\overline{\Psi} _{SO}-\overline{\Psi} _{TW} \right) | _{\theta=\theta_c}(z).
$$

The model is integrated in time until a steady state is reached.
