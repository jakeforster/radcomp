from sympy import Function, Eq, exp, dsolve, solve
from sympy.abc import t

y1 = Function("y1")(t)
y2 = Function("y2")(t)

# Define the ODEs
ode1 = Eq(
    y1.diff(t), -0.4 * y1 + 0.09 * (4e10 * exp(-1.0 * t) + 3.2e10 * exp(-0.1 * t))
)
ode2 = Eq(y2.diff(t), -1.2 * y2 + 0.09 * (-4e10 * exp(-1.0 * t) + 4e10 * exp(-0.1 * t)))


# Solve the ODEs
sol = dsolve((ode1, ode2), [y1, y2])

# Initial conditions
y1_0 = 1.8e10
y2_0 = 9e9

# Solve for the constants
constants = solve(
    (
        sol[0].rhs.subs(t, 0) - y1_0,
        sol[1].rhs.subs(t, 0) - y2_0,
    )
)

# Substitute the constants back into the solution
y1_sol = sol[0].rhs.subs(constants)
y2_sol = sol[1].rhs.subs(constants)

print(y1_sol)
print(y2_sol)
