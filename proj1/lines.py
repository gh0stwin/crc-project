import numpy as np 
import matplotlib.pyplot as plt

axes = plt.subplot()
color = ["b","g","r","c","m","y"]
N_values = [10, 10**2, 10**3, 10**4, 10**5, 10**6]
N_text = [r"$10$",r"$10^2$",r"$10^3$",r"$10^4$",r"$10^5$",r"$10^6$"]
N_print = ["10","10^2","10^3","10^4","10^5","10^6"]
for c in range(len(N_values)):
    N = N_values[c]
    ba = np.loadtxt(
        './results/clustering_coefficient/' + str(N) + '-ba.out',
        delimiter=',',
        dtype="float"
    )
    dms = np.loadtxt(
        './results/clustering_coefficient/' + str(N) + '-dms.out',
        delimiter=',',
        dtype="float"
    )
    # process data
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
            degreesBA.append(j+1)
            ba_avg_ck.append(sum(auxBA)/len(auxBA))
        if len(auxDMS) != 0:
            degreesDMS.append(j+1)
            dms_avg_ck.append(sum(auxDMS)/len(auxDMS))

    degreesBA = np.asarray(degreesBA)
    ba_avg_ck = np.asarray(ba_avg_ck)
    degreesDMS = np.asarray(degreesDMS)
    dms_avg_ck = np.asarray(dms_avg_ck)

    ## LINE BA
    # fit ba points
    trendBA = np.polyfit(np.log(degreesBA), np.log(ba_avg_ck),1)
    trendpolyBA = np.poly1d(trendBA) 
    
    #axes.loglog(degreesBA, ba_avg_ck, 'ro', color='red', markersize=3, label="BA")
    yfit = lambda x: np.exp(trendpolyBA(np.log(x)))
    #x = range(1,N_values[len(N_values)-1])
    #y = yfit(x)
    y = yfit(degreesBA)

    X = degreesBA - degreesBA.mean()
    Y = y - y.mean()

    slope = (X.dot(Y)) / (X.dot(X))
    print(N_print[c] + ' slopes:')
    print('\tBA:  ' + str(slope))
    plt.loglog(degreesBA, y, linewidth=2, label="BA line fit for " + N_text[c], color=color[c])
    #if c == len(N_values)-1:
    #    axes.loglog(degreesDMS, dms_avg_ck, 'ro', color='blue', markersize=3, label="DMS 10^6")
    #print(str(N) + ' done')

    ## LINE DMS
    # fit ba points
    trendDMS = np.polyfit(np.log(degreesDMS), np.log(dms_avg_ck),1)
    trendpolyDMS = np.poly1d(trendDMS) 
    
    #axes.loglog(degreesDMS, ba_avg_ck, 'ro', color='red', markersize=3, label="BA")
    yfit = lambda x: np.exp(trendpolyDMS(np.log(x)))
    #x = range(1,N_values[len(N_values)-1])
    #y = yfit(x)
    y = yfit(degreesDMS)

    X = degreesDMS - degreesDMS.mean()
    Y = y - y.mean()

    slope = (X.dot(Y)) / (X.dot(X))
    print('\tDMS: ' + str(slope))
    plt.loglog(degreesDMS, y, '--', linewidth=2, label="DMS line fit for " + N_text[c], color=color[c])
    

axes.legend(loc="upper right")
plt.xlabel(r"$k$")
plt.ylabel(r"$<C_k>$")
plt.show()