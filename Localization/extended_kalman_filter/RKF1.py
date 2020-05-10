"""

Extended kalman filter (EKF) localization sample

author: Atsushi Sakai (@Atsushi_twi)

"""

import math

import matplotlib.pyplot as plt
import numpy as np

# Covariance for EKF simulation
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance
R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

#  Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
# INPUT_MORE_NOISE = np.diag([2.0, np.deg2rad(40.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2
GPS_MORE_NOISE = np.diag([2.0, 2.0]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]

show_animation = True


def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v], [yawrate]])
    return u


def observation(xTrue, xd, u, i):
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    if i % 15 == 0:
        z = observation_model(xTrue) + GPS_MORE_NOISE@np.random.randn(2, 1)
    else:
        z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)

    # add noise to input
    if i % 15 == 0:
        ud = u + INPUT_NOISE @ np.random.randn(2, 1)
    else:
        ud = u + INPUT_NOISE @ np.random.randn(2, 1)

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud


def motion_model(x, u):
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F @ x + B @ u

    return x


def observation_model(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    z = H @ x

    return z


def jacob_f(x, u):
    """
    Jacobian of Motion Model

    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF


def jacob_h():
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    return jH


def ekf_estimation(xEst, PEst, z, u):
    #  Predict
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q

    #  Update
    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    return xEst, PEst


# alpha is quantile limit for outlier detection of Chi-square distribution of Maha distant
def rkf_estimation(jH, Q, R, z, u, xEst, PEst, alpha=0.9):
    #  Predict
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q
    PEst = PPred

    xEst = xPred

    #De-correlation
    L = np.linalg.cholesky(R)
    Linv = np.linalg.inv(L)
    # zPred = observation_model(xPred) # y_k
    # zPRedDeCor = Linv@zPred #\bar(y_k)
    zDeCor = Linv@z
    # print(z, " and ", zDeCor)
    jHDeCor = Linv@jH 

    #Sequential update
    m = max(z.shape)
    quantileChiSquare=2.706#1.5#2.706
    remainIdx = {i for i in range(m)}

    for j in range(m):

        smallestOne = None
        minMahaDist = 100
        choosedZ = None
        choosedIdx = None
        choosedH = None
        for i in remainIdx:
            h = jHDeCor[i] 
            Yki = h@xEst
            COVki = h@PEst@h.T+1
            mahaDist = (zDeCor[i]-Yki)**2/COVki

            if minMahaDist > mahaDist[0]:
                minMahaDist = mahaDist[0]
                smallestOne = [Yki, COVki, mahaDist[0]]
                choosedZ = zDeCor[i]
                choosedIdx = i
                choosedH = h
                print(COVki)
        remainIdx = remainIdx - {choosedIdx}
        k = 1
        # print(smallestOne)
        if smallestOne is None: continue
        if smallestOne[2] > quantileChiSquare:
            k = smallestOne[2]/quantileChiSquare
        
        smallestOne[1] = k*smallestOne[1]
        # update in jth interation
        # K = PEst@jHDeCor[j].T / smallestOne[1]
        K = PEst@choosedH.T / smallestOne[1]
        K = K.reshape(len(K),1)
        # print(K.shape, (choosedZ - smallestOne[0]).shape)
        xEst = xEst + K @ np.array([(choosedZ - smallestOne[0])])
        # xEst = xEst + K *(choosedZ - smallestOne[0])
        # print(K@K.T)
        # print(K.T@K)
        PEst = PEst - smallestOne[1]*K@K.T
        # print(xEst)
    return xEst, PEst

import scipy.linalg
# alpha is quantile limit for outlier detection of Chi-square distribution of Maha distant
def rkf_estimation1(jH, Q, R, z, u, xEst, PEst, alpha=0.9, moment=0.7):
    #  Predict
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q
    PEst = PPred

    xEst = xPred

    #De-correlation
    L = np.linalg.cholesky(R)
    Linv = np.linalg.inv(L)
    # zPred = observation_model(xPred) # y_k
    # zPRedDeCor = Linv@zPred #\bar(y_k)
    zDeCor = Linv@z
    # print(z, " and ", zDeCor)
    jHDeCor = Linv@jH 

    #Sequential update
    m = max(z.shape)
    quantileChiSquare=2.706#2.706
    remainIdx = {i for i in range(m)}

    for j in range(m):

        smallestOne = None
        minMahaDist = 100
        choosedZ = None
        choosedIdx = None
        choosedH = None
        for i in remainIdx:
            h = jHDeCor[i] 
            Yki = h@xEst
            COVki = h@PEst@h.T+1
            mahaDist = (zDeCor[i]-Yki)**2/COVki

            if minMahaDist > mahaDist[0]:
                minMahaDist = mahaDist[0]
                smallestOne = [Yki, COVki, mahaDist[0]]
                choosedZ = zDeCor[i]
                choosedIdx = i
                choosedH = h
                print(COVki)
        remainIdx = remainIdx - {choosedIdx}
        k = 1
        # print(smallestOne)
        if smallestOne is None: continue
        if smallestOne[2] > quantileChiSquare:
            k = smallestOne[2]/quantileChiSquare
            quantileChiSquare += moment
        else:
            quantileChiSquare -= moment
        
        smallestOne[1] = k*smallestOne[1]
        # update in jth interation
        # K = PEst@jHDeCor[j].T / smallestOne[1]
        K = PEst@choosedH.T / smallestOne[1]
        K = K.reshape(len(K),1)
        # print(K.shape, (choosedZ - smallestOne[0]).shape)
        xEst = xEst + K @ np.array([(choosedZ - smallestOne[0])])
        # xEst = xEst + K *(choosedZ - smallestOne[0])
        # print(K@K.T)
        # print(K.T@K)
        PEst = PEst - smallestOne[1]*K@K.T
        # print(xEst)
    return xEst, PEst  

def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
    rot = np.array([[math.cos(angle), math.sin(angle)],
                    [-math.sin(angle), math.cos(angle)]])
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")

import scipy.spatial.distance as dst
def comuteError(hxEst, hxTrue):
    s = 0
    hxEst = np.array(hxEst).T
    hxTrue = np.array(hxTrue).T
    print("len : ", len(hxEst), len(hxTrue))
    for xEst, xTrue in zip(hxEst, hxTrue):
        # print(np.array(xEst), np.array(xTrue))
        s += dst.euclidean(np.array(xEst), np.array(xTrue))
    return s

def main(name_save):
    print(__file__ + " start!!")

    time = 0.0

    # State Vector [x y yaw v]'
    xEst = np.zeros((4, 1))
    xEstRKF1 = np.zeros((4, 1))
    xEstRKF = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst =  np.eye(4)
    PEstRKF =  np.eye(4)
    PEstRKF1 =  np.eye(4)

    xDR = np.zeros((4, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxEstRKF = xEstRKF
    hxEstRKF1 = xEstRKF1
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))

    i = 0

    while SIM_TIME >= time:
        time += DT
        i += 1
        u = calc_input()

        xTrue, z, xDR, ud = observation(xTrue, xDR, u, i)
        xEstRKF, PEstRKF = rkf_estimation(jH=jacob_h(), Q=Q, R=R, z=z, u=ud, xEst=xEstRKF, PEst=PEstRKF, alpha=0.9)

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)
        xEstRKF1, PEstRKF1 = rkf_estimation1(jH=jacob_h(), Q=Q, R=R, z=z, u=ud, xEst=xEstRKF1, PEst=PEstRKF1, alpha=0.9)

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxEstRKF = np.hstack((hxEstRKF, xEstRKF))
        hxEstRKF1 = np.hstack((hxEstRKF1, xEstRKF1))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

        if not show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b")
            plt.plot(hxDR[0, :].flatten(),
                     hxDR[1, :].flatten(), "-k")
            plt.plot(hxEstRKF[0, :].flatten(),
                     hxEstRKF[1, :].flatten(), "-m")
            plt.plot(hxEstRKF1[0, :].flatten(),
                     hxEstRKF1[1, :].flatten(), "-r")       
            # plt.plot(hxEst[0, :].flatten(),
            #          hxEst[1, :].flatten(), "-r")
            plot_covariance_ellipse(xEst, PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)
    # print("EKF's error: ",comuteError(xEst, xTrue))
    # print("RKF's error: ",comuteError(xEstRKF, xTrue))
    # print("RKF1's error: ",comuteError(xEstRKF1, xTrue))
    plt.savefig("{}.png".format(name_save))
    return comuteError(hxEst, hxTrue), comuteError(hxEstRKF, hxTrue), comuteError(hxEstRKF1, hxTrue)


if __name__ == '__main__':
    hekf = []
    hrkf = []
    hrkf1 = []
    for i in range(20):
        ekf, rkf, rkf1 = main(i)
        hekf.append(ekf)
        hrkf.append(rkf)
        hrkf1.append(rkf1)
    print("EKF's average: ", sum(hekf)/len(hekf))
    print("RKF's average: ", sum(hrkf)/len(hrkf))
    print("RKF1's average: ", sum(hrkf1)/len(hrkf1))
