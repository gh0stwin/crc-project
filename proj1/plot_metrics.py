def plot_graph_metrics(f_name, cols):
    f = open(f_name, 'r')
    line = f.readline()

    while line:
        all_values = line.split(',')
        values = []

        for i in cols:
            values.append(float(all_values[i]))
            

        line = f.readline()

def avg_metrics_by_n(f, cols):
    line = f.readline()
    n = line.split(',')[0].split('-')[1]
    new_n = n

    while True:
        line = f.readline()
