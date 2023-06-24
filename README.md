[![DOI](https://zenodo.org/badge/634282475.svg)](https://zenodo.org/badge/latestdoi/634282475)

# Radcomp: Radioactive compartment models

Radcomp is a Python package for modelling the movement of radioactive nuclei and their progeny through a system. 

Currently only deterministic compartment models are provided. 

## Requires

Python >= 3.10, < 3.12

## Installation

    python -m pip install radcomp

## Deterministic compartment model

A system of $m$ "layers" and $n$ compartments is modelled, where each layer corresponds to a different nuclide and all layers share the same set of compartments. 

The deterministic compartment model (DCM) is specified by $m \times n$ ODEs of the form:

$\frac{dN_{ai}}{dt} = \sum_j M_{aij} N_{aj} - N_{ai} \left( \lambda_a + \sum_j M_{aji}  \right) + \sum_{b < a} f_{ab} \lambda_b W_{bi}(t)$,

where
+ $N_{ai}$ is the number of nuclei in layer $a$ compartment $i$. This is solved numerically on a grid of times. 
+ $M_{aij}$ is the coefficient of transfer of the nuclide in layer $a$ from compartment $j$ to $i$. Note $M$ is an $m \times n \times n$ array with $M_{aij} \geq 0$ and $M_{aii} = 0$.
+ $\lambda_a$ is the transition rate (physical decay constant) of the nuclide in layer $a$. Must be $\geq 0$.
+ $f_{ab}$ is the branching fraction (0 to 1) for the nuclide in layer $b$ to the nuclide in layer $a$. Note $f$ is an $m \times m$ array with $0 \leq f_{ab} \leq 1$ and $f_{ab} = 0$ for $b \geq a$.
+ $W_{bi}(t)$ is the number of nuclei in layer $b$ compartment $i$ at time $t$. This is a function defined for all times in the integration interval (not just on the grid) and returns values $\geq 0$. Here it is approximated by linear interpolation of the solution for $N_{bi}$ (except for the prelayer, if one is used - see [below](#option-to-input-time-activity-curves-in-a-prelayer)).

The terms on the RHS of the equation above represent:
+ transfer into the compartment 
+ physical decay
+ transfer out of the compartment
+ growth by decay of parent nuclei

This is an initial value problem; i.e., provided the initial values $N_{ai}(0)$ it can be solved by numerical integration. 
Each layer $a$ (described by a system of $n$ ODEs) is solved separately and in order, with the solution for one layer able to serve as input for subsequent layers. 

This program aims to provide a convenient interface for solving the DCM for any system of layers and compartments.

## TOML configuration file

Input parameters to models are read from a TOML file. 
For example, the method `solve_dcm_from_toml()` for solving a DCM requires the filepath to a TOML file and an array of times at which the solution will be found.
In the TOML file, the parameters for each layer are specified separately.
The parameters for a layer are provided below a line containing `[[layer]]`. 
The layers must be provided in an order such that the nuclide in one layer cannot transition to any of the nuclides in previous layers. 
The "keys" to specify a layer depend on the type of model and are described below.

There is also the option to assign names to the compartments in the TOML file. 
To do this, include a line containing `[compartments]` and below it assign the key `names` to an array of strings, which will be the names of the compartments in order.
E.g. 

``` toml
    [compartments]
    names = ["plasma", "kidneys", "lungs"]
```

## Layer keys for a DCM

The model parameters for a DCM are $M_{aij}$, $\lambda_a$, $f_{ab}$, and $N_{ai}(0)$.
Valid keys that can be used to specify these parameters for each layer $a$ are given in [Table 1](#table-dcm-layer-keys).

<a name="table-dcm-layer-keys"></a>

| Key               | Description                                                | Units          | Associated model parameter                           | Data type (size)                              | Required?                                                    |
|-------------------|------------------------------------------------------------|----------------|------------------------------------------------------|-----------------------------------------------|--------------------------------------------------------------|
| `xfer_coeffs_h-1` | Transfer coefficients between compartments for layer       | h<sup>-1</sup> | $M_{aij}$ for $i,j = 1,\ldots,n$                     | Array of $n$ Arrays of $n$ Floats or Integers | If $n > 1$ (else ignored)                                    |
| `trans_rate_h-1`  | Transition rate for (nuclide in) layer                     | h<sup>-1</sup> | $\lambda_a$                                          | Float or Integer                              | Yes                                                          |
| `branching_fracs` | Branching fractions (0 to 1) of layer to subsequent layers |                | $f_{ba}$ for $b = a+1,\ldots, m$                     | Array of $m-a$ Floats or Integers             | If $m-a \geq 1$ and `trans_rate_h-1` $\neq 0$ (else ignored) |
| `initial_MBq`     | Initial activity of nuclide in layer in each compartment   | MBq            | $A_{ai}(0) = \lambda_a N_{ai}(0)$ for $i=1,\ldots,n$ | Array of $n$ Floats or Integers               | If `trans_rate_h-1` $\neq 0$ (else ignored)                  |
| `initial_nuclei`  | Initial number of nuclei in layer in each compartment      |                | $N_{ai}(0)$ for $i=1,\ldots,n$                       | Array of $n$ Floats or Integers               | If `trans_rate_h-1` $= 0$ (else ignored)                     |
| `name`            | Name of layer                                              |                |                                                      | String                                        | No                                                           |

Table 1: Layer keys for a DCM. Here $a$ is the index of the layer (1-based index), $m$ is the number of layers, and $n$ is the number of compartments.

Note:
+ The values in `xfer_coeffs_h-1`, `initial_MBq`, and `initial_nuclei` are in order of compartments.
+ The first value in `branching_fracs` is for the transition to the layer immediately below. 
To check the input was as intended, call the `info_xfer()` and `info_growth()` methods of the `DetCompModelSol` instance.

An example of a TOML file for a DCM is provided below. 

``` toml
[compartments]
names = ["Organ 1", "Organ 2", "Organ 3"]

[[layer]]
name = "Nuclide A"
trans_rate_h-1 = 0.91
branching_fracs = [0.1, 0.9]
initial_MBq = [1, 3.3, 2]
xfer_coeffs_h-1 = [
[0, 3.4, 2.1],  # flow to Organ 1
[4.3, 0, 1.3],  # flow to Organ 2
[5.8, 9.2, 0]   # flow to Organ 3
]

[[layer]]
name = "Nuclide B"
trans_rate_h-1 = 0  # stable
initial_nuclei = [3, 2, 2]
xfer_coeffs_h-1 = [[0, 3, 2], [4, 0, 1], [5.1, 9, 0]]

[[layer]]
name = "Nuclide C"
trans_rate_h-1 = 0.2
initial_MBq = [0, 1.3, 0]
xfer_coeffs_h-1 = [[0, 1, 0], [2, 0, 1.3], [3, 1, 0]]
```

Note in this example, the transfer coefficient for Nuclide A from Organ 2 to Organ 3 is 9.2 h<sup>-1</sup> and the branching fraction of Nuclide A to Nuclide C is 0.9.


## Option to input time activity curves in a prelayer

The user has the option to supply time-activity curves (TACs) for a nuclide that is able to transition to one or more layers in the model. 
The TACs in the "prelayer" can then contribute to the growth of nuclei in layers. 
If a prelayer is provided, prelayer TACs must be provided for all compartments. 

Unlike the layers in the model, the prelayer is **not** specified in the input TOML file. 
Instead, pass an instance of the `Prelayer` class to the instantiating model method (e.g. `solve_dcm_from_toml()`) using the optional keyword argument `prelayer`.


## Voiding
(New in Version 0.1.0)

The user has the option to void nuclei from compartments at certain times.
This is specified by creating voiding rules.
Create one or more instances of the `VoidingRule` class and pass them to the instantiating model method (e.g. `solve_dcm_from_toml()`) using the optional keyword argument `voiding_rules`.
See [examples](README.md#examples).
The number of nuclei and activity voided is recorded in the `DetCompModelSol` instance.

## API reference

https://radcomp.readthedocs.io

## Examples {#examples}

https://github.com/jakeforster/radcomp/tree/main/examples

## Testing

    python -m pytest

## TODO

Add stochastic models

