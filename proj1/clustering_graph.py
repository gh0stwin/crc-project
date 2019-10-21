import numpy as np 
import matplotlib.pyplot as plt
import argparse
import os
from scipy.optimize import curve_fit


def main():
    # Deal with arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'N', 
        action="store",
        type=int, 
        help='N for number nodes for BA and DMS \
            to load clustering coefficient files'
    )
    parser.add_argument(
        'show_all_BA_values', 
        action="store",
        nargs='?',
        default=False,
        type=bool, 
        help='boolean to show all BA clustering coefficient values'
    )
    args = parser.parse_args()

    N = args.N
    ba_file = './results/clustering_coefficient/' + str(N) + '-ba.out'
    dms_file = './results/clustering_coefficient/' + str(N) + '-dms.out'
    if not os.path.isfile(ba_file) or not os.path.isfile(dms_file):
        print('Files for N=' + str(N) + ' do not exist' )
        exit()
    
    ba = np.loadtxt(
        ba_file,
        delimiter=',',
        dtype="float"
    )
    dms = np.loadtxt(
        dms_file,
        delimiter=',',
        dtype="float"
    )

    degreesBA = []
    ba_avg_ck = []
    degreesDMS = []
    dms_avg_ck = []
    for j in range(ba.shape[1]):
        auxBA = []
        auxDMS = []
        for i in range(ba.shape[0]):
            if ba[i,j] != 0:
                auxBA.append(ba[i,j])
            if dms[i,j] != 0:
                auxDMS.append(dms[i,j])
        if len(auxBA) != 0:
            degreesBA.append(j)
            ba_avg_ck.append(sum(auxBA)/len(auxBA))
        if len(auxDMS) != 0:
            degreesDMS.append(j)
            dms_avg_ck.append(sum(auxDMS)/len(auxDMS))

    degreesBA = np.asarray(degreesBA)
    ba_avg_ck = np.asarray(ba_avg_ck)
    degreesDMS = np.asarray(degreesDMS)
    dms_avg_ck = np.asarray(dms_avg_ck)

    # BA line fit
    trendBA = np.polyfit(np.log(degreesBA), np.log(ba_avg_ck),1)
    trendpolyBA = np.poly1d(trendBA) 

    fig, axes = plt.subplots(figsize=(3,3))

    # DMS
    axes.loglog(degreesDMS, dms_avg_ck, 'o', mfc='none', color='black', markersize=2, label="DMS")

    #BA all values
    if args.show_all_BA_values:
        x = []
        y = []

        for j in range(ba.shape[1]):
            for i in range(ba.shape[0]):
                if ba[i,j] != 0:
                    x.append(j)
                    y.append(ba[i,j])

        plt.loglog(x, y, 'o', color='blue', markersize=2, 
            label="all BA clustering coefficient values")

    axes.loglog(degreesBA, ba_avg_ck, 'o', color='black', markersize=2, label="BA")
    
    # BA
    yfit = lambda x: np.exp(trendpolyBA(np.log(x)))
    y = yfit(degreesBA)

    X = degreesBA - degreesBA.mean()
    Y = y - y.mean()

    slope = (X.dot(Y)) / (X.dot(X))
    print('BA slope ' + str(slope))
    
    # DMS
    # Powerlaw 
    def f(x, N, a):
        return N*np.power(x,-a)

    popt, pcov = curve_fit(f, degreesDMS, dms_avg_ck)
    print('DMS: %1.2fk^-%1.2f' %(popt[0], popt[1]))

    plt.loglog(degreesBA, y, color="red", linestyle='-', linewidth=2, label="BA line fit")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$<C_k>$")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
