import pandas as pd
import numpy as np

# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------

seed = 42
np.random.seed(seed)

#-------------------------------------------------WHOLE-------------------------------------------------
file = open("/hps/research1/icortes/acunha/python_scripts/Molecular_vae/loss_results.txt","r")
file = file.readlines()

validation_loss_total = []
validation_loss_recon = []
validation_loss_kl= []
test_loss_total = []
test_loss_recon = []
test_loss_kl = []
train_loss_total = []
train_loss_recon = []
train_loss_kl = []
loss_params = []
valid = []
alpha = []

i = 0
while i < len(file):
    if "Training" in file[i] and "Validation" in file[i+1] and "Testing" in file[i+2]:
        train_line = file[i].strip("\n").split(' ; ')
        validation_line = file[i+1].strip("\n").split(' ; ')
        test_line = file[i+2].strip("\n").split(' ; ')
        parameters = file[i+4].strip(".txt\n").split("/")
        valid_line = file[i+3].strip('%\n').split(': ')
        
        validation_loss_total.append(float(validation_line[0].split(':')[-1]))
        validation_loss_recon.append(float(validation_line[1].split(':')[-1]))
        validation_loss_kl.append(float(validation_line[2].split(':')[-1]))
        train_loss_total.append(float(train_line[0].split(':')[-1]))
        train_loss_recon.append(float(train_line[1].split(':')[-1]))
        train_loss_kl.append(float(train_line[2].split(':')[-1]))
        test_loss_total.append(float(test_line[0].split(':')[-1]))
        test_loss_recon.append(float(test_line[1].split(':')[-1]))
        test_loss_kl.append(float(test_line[2].split(':')[-1]))
        loss_params.append(parameters[-1])
        alpha.append(parameters[1])
        valid.append(float(valid_line[-1]))
        
        i += 5
    
    else:
        with open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/check_cases.txt', 'a') as f:
            parameters = file[i].split("put_")[-1].strip(".txt\n").split("_")
            f.write("_".join(parameters))
            f.write('\n')
        i += 1
d = pd.DataFrame(validation_loss_total, columns = ['Val_loss_total'])
d['Val_loss_recon'] = validation_loss_recon
d['Val_loss_kl'] = validation_loss_kl
d['Train_loss_total'] = train_loss_total
d['Train_loss_recon'] = train_loss_recon
d['Train_loss_kl'] = train_loss_kl
d['Test_loss_total'] = test_loss_total
d['Test_loss_recon'] = test_loss_recon
d['Test_loss_kl'] = test_loss_kl
d['Valid_molecules'] = valid
d['Alpha'] = alpha
d['Difference'] = np.abs(d['Train_loss_total'] - d['Val_loss_total'])
d['Parameters'] = loss_params
print(d)
d = d.sort_values(['Valid_molecules'], ascending = False)

best_parameters = d.head(20)
# best_parameters.to_csv("/hps/research1/icortes/acunha/python_scripts/single_cell/best_parameters_pancancer_losses.txt")
# best_parameters = best_parameters.sort_values('Difference')
print(best_parameters.head(20))
print(list(best_parameters['Parameters'].head(20)))

with open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/list_best_parameters.txt', 'w') as f:
    f.write('{}/{}'.format(list(best_parameters['Alpha'].head(1))[0],
                           list(best_parameters['Parameters'].head(1))[0].strip('output_')))
    f.write('\n')
    f.write('{}/{}'.format(list(best_parameters['Alpha'].head(1))[0],
                           list(best_parameters['Parameters'].head(1))[0]))