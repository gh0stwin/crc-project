import math
import networkx as nx
import os
import pathlib as pl
import traceback

from evaluate_results import _str_to_float
from graph_generator import configuration_model, dms
from sir.sir_controller import SirController


def get_files_by_props(
    models, 
    nodes, 
    min_idx, 
    max_idx, 
    path='./networks/'
):
    folder = pl.Path(path)
    matched_files = []

    for file in folder.iterdir():
        if file.suffix != '.gml':
            continue

        file_model, file_nodes, file_iter = file.stem.split('_')[0:3]

        if (
            file_model in models and
            int(file_nodes) in nodes and 
            min_idx <= int(file_iter) < max_idx
        ):
            matched_files.append(str(file))
    
    return matched_files

def check_files_with_no_results(
    matched_files, 
    betas,
    fs,
    results_path='./results/'
):
    miss_res = {}
    res_files = [
        f for f in pl.Path(results_path).iterdir() if f.suffix == '.out'
    ]

    for file in matched_files:
        miss_res[file] = {b:list(fs) for b in betas}
        f_model, f_nodes, f_iter = pl.Path(file).stem.split('_')[0:3]

        for b in betas:
            for f in fs:
                for res_f in res_files:
                    aux = res_f.stem.split('_')
                    res_f_model = aux[0]
                    res_f_nodes = aux[1]
                    res_f_iter = aux[2]
                    res_f_beta = _str_to_float(aux[3])
                    res_f_f = _str_to_float(aux[4])

                    if (
                        math.isclose(res_f_beta, b) and
                        math.isclose(res_f_f, f) and
                        res_f_model == f_model and
                        res_f_nodes == f_nodes and
                        res_f_iter == f_iter
                    ):
                        for i in range(len(miss_res[file][b])):
                            if math.isclose(f, miss_res[file][b][i]):
                                miss_res[file][b].pop(i)
                                break

                        if len(miss_res[file][b]) == 0:
                            del miss_res[file][b]
                    
    return miss_res

def get_result_missing_files(
    models, 
    ns, 
    betas, 
    net_idxs, 
    fs, 
    path='./results/'
):
    results_files = [f.stem for f in pl.Path(path).glob('*.out')]

    for model in models:
        for n in ns:
            for net_idx in net_idxs:
                for beta in betas:
                    for f in fs:
                        f_name = '{}_{}_{}_{}_{}'.format(
                            model,
                            n,
                            net_idx,
                            beta.replace('.', ''),
                            f.replace('.', '')
                        )

                        if f_name not in results_files:
                            print(f_name)

def run_files_with_miss_results(miss_results, iters, res_path):
    sh = SirController(path=res_path)

    for file in miss_results:
        for beta in miss_results[file]:
            sh.simulate(
                [file], 
                [beta],
                miss_results[file][beta],
                iters,
                0
            )

def parse_results(files):
    for file in files:
        # print(str(file))
        f = open(file, 'r')
        all_values = f.readline().split(',')
        f.close()

        values = []
        line = []
        i = 0
        seed = None

        try:
            while i < len(all_values):
                for _ in range(2):
                    if i == 0:
                        seed = int(all_values[0]) + 1

                    line.append(all_values[i])
                    i += 1

                line.append(all_values[i])
                if all_values[i] == 'RW' or all_values[i] == 'TS':
                    l = 1
                else:
                    l = 0

                i += 1

                for _ in range(l):
                    line.append(all_values[i])
                    i += 1

                if i == len(all_values) - 1:
                    line.append(all_values[i])
                    values.append(line)
                    break
                else:
                    line.append(all_values[i][:-len(str(seed))])
                    seed = int(all_values[i][-len(str(seed)):]) + 1
                    all_values[i] = str(seed - 1)

                values.append(line)
                line = []
        except Exception:
            # traceback.print_exc()
            print('error: ' + str(file))
            # print(i)
            continue

        f = open(file, 'w')

        for line in values:
            f.write(line[0])

            for i in range(len(line)):
                if i == 0:
                    continue

                f.write(',' + line[i])

            f.write('\n')

        f.close()
        
def run_sir(
    models, 
    ns, 
    betas, 
    fs, 
    iters, 
    min_idx, 
    max_idx,
    seed,
    net_path='./networks/', 
    res_path='./results/'
):
    files = get_files_by_props(models, ns, min_idx, max_idx, net_path)
    sh = SirController(res_path)
    sh.simulate(files, betas, fs, iters, seed)
    print('\ndone!')

if __name__ == '__main__':
    # create_networks(
    #     [625, 1250, 2500, 5000, 10000],
    #     configuration_model,
    #     (2.5, lambda n, g: int(round(n ** (1 / (g - 1))))),
    #     './networks/configuration-model',
    #     300,
    #     0
    # )

    # create_networks(
    #     [625, 1250, 2500, 5000, 10000],
    #     dms,
    #     tuple(),
    #     './networks/dms',
    #     300,
    #     0
    # )

    # create_networks(
    #     [625, 1250, 2500, 5000, 10000],
    #     nx.barabasi_albert_graph,
    #     (2,),
    #     './networks/ba',
    #     300,
    #     0
    # )

    ####################################################################

    # files = sorted(pl.Path('.').glob('**/*.gml'))
    # betas = [0.5, 1, 2]
    # fs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # iters = 40
    # sh = SirHandler()
    # sh.simulate(files, betas, fs, iters, 0)

    ####################################################################

    # networks_path = './networks/'
    # results_path = './results/'
    # betas = [0.5, 1, 2]
    # fs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # miss_res = check_files_with_no_results(
    #     get_files_by_props(['ba'], [10000], 20, 21, networks_path),
    #     betas,
    #     fs,
    #     results_path
    # )

    # print(miss_res)

    ####################################################################

    files = sorted(pl.Path('.').glob('./results/*.out'))
    parse_results(files)

    ####################################################################

    # networks_path = './networks/'
    # results_path = './results/'
    # betas = [2, 4, 8]
    # fs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # iters = 1
    # files = get_files_by_props(
    #     ['configuration-model'], 
    #     [625], 
    #     1617, 
    #     101, 
    #     networks_path
    # )

    # sh = SirController(results_path)
    # sh.simulate(files, betas, fs, iters, 0)
