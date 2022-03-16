import numpy as np
from scipy.signal import dlsim, dlti, lsim, lti
from scipy.linalg import solve_discrete_are, expm, solve_continuous_are
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import torch

# c, params = initialize_plot('Presentation')
# plt.rcParams.update(params)

PLOT_TRAJECTORY = False
DISCRETE = True
USE_DGL = False


# function that returns dz/dt
def inv_pendulum(state, t, u, params):
    x, dx, phi, dphi = state

    # get params
    T1 = params['T1']
    K = params['K']
    mp = params['mass_pole']
    l = params['length_pole']
    mu_friction = params['friction_coef']
    g = 9.81
    d = 0.005
    Jd = mp * (l / 2) ** 2 + 1 / 12 * mp * l ** 2 + 0.25 * mp * (d / 2) ** 2

    # set up odes
    xdt = dx
    xdotdt = 1 / T1 * (K * u - dx)
    phidt = dphi
    phidotdt = 0.5 * mp * g * l / Jd * np.sin(phi) \
               - 0.5 * mp * l / Jd * np.cos(phi) * 1 / T1 * (K * u - dx) \
               - mu_friction / Jd * dphi
    dxdt = [xdt, xdotdt, phidt, phidotdt]
    return dxdt


def get_lqr_cost(x, u, Q, R, eval_logcost=True):
    # get size
    dim_x = 4
    dim_u = 1

    x = x.reshape(-1, 1, dim_x)
    x_T = x.reshape(-1, dim_x, 1)
    u = u.reshape(-1, 1, dim_u)
    u_T = u.reshape(-1, dim_u, 1)
    cost = x @ Q @ x_T + u @ R @ u_T

    time_steps = u.shape[0]
    cost = np.sum(cost) / time_steps
    return cost


def get_opt_state_controller(model, Q, R):
    (Ad, Bd, A, B) = model
    # solve ricatti equation
    if DISCRETE:
        P = solve_discrete_are(Ad, Bd, Q, R)
        # calculate optimal controller gain
        K = np.linalg.inv(Bd.T @ P @ Bd + R) @ Bd.T @ P @ Ad
    else:
        P = solve_continuous_are(A, B, Q, R)
        # calculate optimal controller gain
        K = np.linalg.inv(R) @ B.T @ P
    return K


def simulate_system_DGL(model, K, parameter, sim_time, Ts):
    # initial condition
    z0 = [4, 0, 0.1, -0.01]

    # number of time points

    # time points
    t = np.arange(0, sim_time, Ts)
    n = int(sim_time / Ts)

    # step input
    u = np.zeros_like(t)
    # change to 2.0 at time = 5.0

    # store solution
    x1 = np.empty_like(t)
    x2 = np.empty_like(t)
    x3 = np.empty_like(t)
    x4 = np.empty_like(t)

    # record initial conditions
    x1[0] = z0[0]
    x2[0] = z0[1]
    x3[0] = z0[2]
    x4[0] = z0[3]

    # solve ODE
    for i in range(1, n):
        # span for next time step
        tspan = [t[i - 1], t[i]]
        # solve for next step
        z = odeint(model, z0, tspan, args=(u[i], parameter))
        # store solution for plotting
        x1[i] = z[1][0]
        x2[i] = z[1][1]
        x3[i] = z[1][2]
        x4[i] = z[1][3]
        # next initial condition
        z0 = z[1]
        z0 += np.random.normal(loc=0, scale=0.005, size=(len(z0),))
        if i < n - 1:
            # next control input
            u[i + 1] = -K @ z0

    if PLOT_TRAJECTORY:
        # plot results
        plt.plot(t, u, 'g:', label='$u(t)$')
        plt.plot(t, x1, 'b-', label='$x(t)$')
        plt.plot(t, x2, 'r--', label='$\\dot{x}(t)$')
        plt.plot(t, x3, 'k-', label='$\\phi(t)$')
        plt.plot(t, x4, 'y--', label='$\\dot{\\phi}(t)$')
        plt.ylabel('-')
        plt.title('States, Control')
        plt.xlabel('Time [$s$]')
        plt.legend()
        plt.show()
        plt.close()

    return np.array([x1, x2, x3, x4]), u


def get_linearized_model(parameter, Ts):
    # get params
    T1 = parameter['T1']
    K = parameter['K']
    mu_friction = parameter['friction_coef']
    mp = parameter['mass_pole']
    l = parameter['length_pole']
    g = 9.81
    d = 0.005

    # state matrix
    J = mp * (l / 2) ** 2 + 1 / 12 * mp * l ** 2 + 0.25 * mp * (d / 2) ** 2
    mpl_J = mp * l / J
    A = np.array([[0, 1, 0, 0],
                  [0, -1 / T1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0.5 * mpl_J / T1, 0.5 * mpl_J * g, -mu_friction / J]])

    # input matrix
    B = np.array([[0], [K / T1], [0], [-0.5 * mpl_J * K / T1]])

    # # from scipy.signal.cont2discrete
    # Build an exponential matrix
    em_upper = np.hstack((A, B))

    # Need to stack zeros under the a and b matrices
    em_lower = np.hstack((np.zeros((B.shape[1], A.shape[0])),
                          np.zeros((B.shape[1], B.shape[1]))))

    em = np.vstack((em_upper, em_lower))
    ms = expm(Ts * em)

    # Dispose of the lower rows
    ms = ms[:A.shape[0], :]

    Ad = ms[:, 0:A.shape[1]]
    Bd = ms[:, A.shape[1]:]

    return (Ad, Bd, A, B)


def simulate_system(model, K, parameter, sim_time, Ts, t, noise):
    (Ad, Bd, A, B) = model
    # initial condition
    z0 = [4, 0, 0.1, -0.01]
    sysC = np.eye(4)

    if DISCRETE:
        # feedback
        sysA = Ad - Bd @ K

        # add noise, trick über B, da B nicht mehr für Feedback gebraucht wird
        sysB = np.eye(4)

        sysD = np.zeros_like(sysB)
        sys = dlti(sysA, sysB, sysC, sysD, dt=Ts)

        t_in = np.arange(0, sim_time, Ts)

        # noise input
        if isinstance(t, np.ndarray):
            t = int(t.reshape(-1)[0])
        np.random.seed(t)
        u_noise = np.random.normal(loc=0, scale=0.0006, size=(len(t_in), 4))
        if noise is False:
            u_noise = np.zeros_like(u_noise)  # 0.0006
        (t, y, x) = dlsim(sys, u=u_noise, t=t_in, x0=z0)
    else:
        # feedback
        sysA = A - B @ K

        # add noise, trick über B, da B nicht mehr für Feedback gebraucht wird
        sysB = np.eye(4)
        sysD = np.zeros_like(sysB)
        sys = lti(sysA, sysB, sysC, sysD)

        t_in = np.arange(0, sim_time, Ts)

        # noise input
        u_noise = np.random.normal(loc=0, scale=0.02, size=(len(t_in), 4))
        (t, y, x) = lsim(sys, U=u_noise, T=t_in, X0=z0)
    u = - K @ x.T
    return t, x, u


def get_params(t):
    mass_pole = 0.0804
    length_pole = 0.147
    friction_coef = 2.2e-3
    K_PT1 = 1
    T_PT1 = 1

    # define parameters of physical system
    obj_params = {'mass_pole': mass_pole,
                  'length_pole': length_pole,
                  'friction_coef': friction_coef,
                  'K': K_PT1,
                  'T1': T_PT1}

    # change in the parameters -> time-varying system
    if 50 < t < 100:
        obj_params['friction_coef'] += 2.2e-3 * (- 1.5 * np.cos((np.pi / 50) * (t - 50)) + 1.5)
    elif t >= 100:
        obj_params['friction_coef'] += 2.2e-3 * 3 + 1.1e-3 * np.sin(-(np.pi / 100) * t)

    return obj_params


def perform_simulation(model, controller, t, noise):
    obj_params = get_params(t)
    sample_time = 0.02
    simulation_time = 20

    # define weights
    Q = np.eye(4) * 10
    R = np.eye(1)

    # start algo
    if not USE_DGL:
        sim_time, states, inputs = simulate_system(model, controller, obj_params, simulation_time, sample_time, t,
                                                   noise)
    else:
        states, inputs = simulate_system_DGL(inv_pendulum, controller, obj_params, simulation_time, sample_time)

    if PLOT_TRAJECTORY and not USE_DGL:
        # plot results
        plt.plot(sim_time, inputs.T, 'g:', label='$u(t)$')
        plt.plot(sim_time, states, label=['$x(t)$', '$\\dot{x}(t)$', '$\\phi(t)$', '$\\dot{\\phi}(t)$'])
        plt.ylabel('-')
        plt.title('States, Control')
        plt.xlabel('Time [$s$]')
        plt.legend()
        plt.show()
        plt.close()

    lqr_cost = get_lqr_cost(states, inputs, Q, R)
    return lqr_cost


def lqr_objective_function_2D(x: torch.Tensor, t: torch.Tensor, noise=True) -> torch.Tensor:
    Q = np.eye(4) * 10
    R = np.eye(1)
    sample_time = 0.02
    fxts = []
    if len(t) < len(x):
        t = torch.ones_like(x)[:, 0] * t

    for xi, ti in zip(x, t):
        controller = xi.numpy()
        test_params = get_params(ti)
        model = get_linearized_model(test_params, sample_time)
        K = get_opt_state_controller(model, Q, R)
        updated_controller = np.append(K[0, 0:2], controller).reshape(1, -1)
        fxt = perform_simulation(model, updated_controller, ti.numpy(), noise)
        fxts.append(fxt)
    fxt = torch.tensor(fxts, dtype=torch.float)
    return fxt


def lqr_objective_function_4D(x: torch.Tensor, t: torch.Tensor, noise=True) -> torch.Tensor:
    sample_time = 0.02
    fxts = []
    if len(t) < len(x):
        t = torch.ones_like(x)[:, 0] * t

    for xi, ti in zip(x, t):
        controller = xi.numpy()
        test_params = get_params(ti)
        model = get_linearized_model(test_params, sample_time)
        updated_controller = controller.reshape(1, -1)
        fxt = perform_simulation(model, updated_controller, ti.numpy(), noise)
        fxts.append(fxt)
    fxt = torch.tensor(fxts, dtype=torch.float)
    return fxt


def lqr_objective_function_3D(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    Q = np.eye(4) * 10
    R = np.eye(1)
    sample_time = 0.02
    fxts = []
    if len(t) < len(x):
        t = torch.ones_like(x)[:, 0] * t

    for xi, ti in zip(x, t):
        controller = xi.numpy()
        test_params = get_params(ti)
        model = get_linearized_model(test_params, sample_time)
        K = get_opt_state_controller(model, Q, R)
        updated_controller = np.append(K[0, 0:2], controller).reshape(1, -1)
        fxt = perform_simulation(model, updated_controller, ti.numpy())
        fxts.append(fxt)
    fxt = torch.tensor(fxts, dtype=torch.float)
    return fxt


if __name__ == '__main__':
    Q = np.eye(4) * 10
    R = np.eye(1)

    time = np.arange(0, 300)
    Ks = []
    og_Ks = []
    costs = []
    costs_ic = []
    for ti in time:
        sample_time = 0.02
        test_params = get_params(ti)
        model = get_linearized_model(test_params, sample_time)

        # calculate optimal controller gain
        K = get_opt_state_controller(model, Q, R)
        og_Ks.append(K.copy())
        # K[0, 0] = theta0
        lqr_cost = perform_simulation(model, K, ti, noise=True)

        Ks.append(K)
        costs.append(lqr_cost)

        # using initial controler
        lqr_cost_ic = perform_simulation(model, Ks[0], ti, noise=True)

        costs_ic.append(lqr_cost_ic)

    og_Ks = np.asarray(og_Ks).reshape(-1, 4)
    Ks = np.asarray(Ks).reshape(-1, 4)
    costs = np.asarray(costs)
    costs_ic = np.asarray(costs_ic)

    plt.figure()
    plt.plot(time, costs / costs[0], label='optimal controler $K_t$')
    plt.plot(time, costs_ic / costs_ic[0], label='inital controler $K_0$')
    plt.ylabel('Normalized cost')
    plt.xlabel('Iteration')
    plt.title('$J(x, u) = \sum_t x^T Q x + R u^2$')
    plt.legend()
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(time, Ks / Ks[0, :], label=['$\\theta_1$', '$\\theta_2$', '$\\theta_3$', '$\\theta_4$', ])
    plt.ylabel('Normalized controler gains')
    plt.xlabel('Iteration')
    plt.title('$K_{LQR} = [\\theta_1, \\theta_2, \\theta_3, \\theta_4]$')
    plt.legend()
    plt.show()
    plt.close()

    thetas0 = np.linspace(-3, -2, 5)
    thetas1 = np.linspace(-6, -4, 5)
    thetas2 = np.linspace(-50, -25, 2)
    thetas3 = np.linspace(-4, -2, 2)

    costs = []
    controllers = []
    t = 150
    test_params = get_params(t)
    model = get_linearized_model(test_params, 0.02)
    # calculate optimal controller gain
    K = get_opt_state_controller(model, Q, R)

    for theta0 in thetas0:
        K[0, 0] = theta0
        for theta1 in thetas1:
            K[0, 1] = theta1
            for theta2 in thetas2:
                K[0, 2] = theta2
                for theta3 in thetas3:
                    K[0, 3] = theta3
                    test_params = get_params(t)
                    model = get_linearized_model(test_params, 0.02)
                    lqr_cost = perform_simulation(model, K, t, noise=True)
                    costs.append(lqr_cost)
                    controllers.append(K.copy())

    costs = np.asarray(costs)
    controllers = np.asarray(controllers).reshape(-1, 4)
    print('done!')
