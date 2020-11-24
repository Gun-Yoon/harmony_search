import numpy as np
import pandas as pd

def fitness_func(list):
    data = pd.read_csv('di_total_data.csv')
    label = data['class']
    predic_label = []
    fit_val = 0

    val_list = ['ecr_i','≥ 254.5','≥ 306.5','1028.5 - 1032.5']
    name_list = ['service','dst_host_srv_count','count','src_bytes']
    hm = [0,0,0,0]
    np_list = np.array(list)
    rule_index = np.where(np_list == 1)[0]

    for i in range(4):
        if list[i] == 1:
            hm[i] = (name_list[i], val_list[i])

    rule_count = 4-(hm.count(0))
    if rule_count == 1:
        for i in range(len(data)):
            if data[hm[rule_index[0]][0]][i] == hm[rule_index[0]][1]:
                predic_label.append('smurf.')
            else:
                predic_label.append('normal.')
    elif rule_count == 2:
        for i in range(len(data)):
            if data[hm[rule_index[0]][0]][i] == hm[rule_index[0]][1] and \
                    data[hm[rule_index[1]][0]][i] == hm[rule_index[1]][1]:
                predic_label.append('smurf.')
            else:
                predic_label.append('normal.')
    elif rule_count == 3:
        for i in range(len(data)):
            if data[hm[rule_index[0]][0]][i] == hm[rule_index[0]][1] and \
                    data[hm[rule_index[1]][0]][i] == hm[rule_index[1]][1] and \
                    data[hm[rule_index[2]][0]][i] == hm[rule_index[2]][1]:
                predic_label.append('smurf.')
            else:
                predic_label.append('normal.')
    elif rule_count == 4:
        for i in range(len(data)):
            if data[hm[rule_index[0]][0]][i] == hm[rule_index[0]][1] and \
                    data[hm[rule_index[1]][0]][i] == hm[rule_index[1]][1] and \
                    data[hm[rule_index[2]][0]][i] == hm[rule_index[2]][1] and \
                    data[hm[rule_index[3]][0]][i] == hm[rule_index[3]][1]:
                predic_label.append('smurf.')
            else:
                predic_label.append('normal.')

    #print(rule_count)
    #print(hm)

    if rule_count == 0:
        return fit_val

    count = 0
    for i in range(len(label)):
        if label[i] == predic_label[i]:
            count = count + 1

    fit_val = (count/len(label))*100

    return fit_val

if __name__ == '__main__':
    # rule test
    list = [1,1,0,0]
    fit = fitness_func(list)
    print(fit)