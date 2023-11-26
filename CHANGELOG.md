# Change Log
All notable changes to this project will be documented in this file.

## [0.0.1] - 2023-04-30

Initial release.

## [0.1.0] - 2023-06-24

Added support for voiding compartments.

## [0.2.0] - 2023-08-30

Relaxed requirements on the versions of dependencies.

## [0.3.0] - 2023-11-26

DCM now uses the "Radau" integration method if there is a transition rate or transfer coefficient greater than 10 inverse hours in the ODE. Such an ODE is "stiff" and the "Radau" method performs much better than the "RK45" method in this case. The "RK45" method is used otherwise.

