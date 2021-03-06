import pandas as pd
import numpy as np

# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------

seed = 42
np.random.seed(seed)

#-------------------------------------------------WHOLE-------------------------------------------------
# file = open("/hps/research1/icortes/acunha/python_scripts/Molecular_vae/loss_results.txt","r")
# file = open("/hps/research1/icortes/acunha/python_scripts/Molecular_vae/loss_results_old.txt","r")
import pandas as pd
import numpy as np

seed = 42
np.random.seed(seed)

check = []
validation_loss_total = []
validation_loss_recon = []
validation_loss_kl= []
validation_valid = []
validation_same = []
test_loss_total = []
test_loss_recon = []
test_loss_kl = []
test_valid = []
test_same = []
train_loss_total = []
train_loss_recon = []
train_loss_kl = []
train_valid = []
train_same = []
loss_params = []


files = open('loss_results_old.txt', 'r')
files = files.readlines()

for file in files:
    values = open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/{}'.format(file.strip('\n')), 'r')
    values = values.readlines()
    try:
        line_train = values[16].strip('\n').split(' :: ')
        line_val = values[17].strip('\n').split(' :: ')
        line_epochs = values[18].strip(' \n').split(': ')
        line_test = values[20].strip('\n').split(' :: ')
        
        assert line_train[0] == 'Training'
        line_train = line_train[1].split(' ; ')
        training_loss = float(line_train[0].split(': ')[1])
        train_recon_loss = float(line_train[1].split(': ')[1])
        train_kl_loss = float(line_train[2].split(': ')[1])
        
        assert line_epochs[0] == 'Number of epochs'
        epoch_stop, epoch_final = line_epochs[1].split(' of ')
        if epoch_stop != epoch_final:
            with open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/stopped_early.txt', 'a') as f:
                f.write(file)
        
        assert line_val[0] == 'Validation'
        line_val = line_val[1].split(' ; ')
        val_loss = float(line_val[0].split(': ')[1])
        val_recon_loss = float(line_val[1].split(': ')[1])
        val_kl_loss = float(line_val[2].split(': ')[1])
        
        assert line_test[0] == 'Testing'
        line_test = line_test[1].split(' ; ')
        test_loss = float(line_test[0].split(': ')[1])
        test_recon_loss = float(line_test[1].split(': ')[1])
        test_kl_loss = float(line_test[2].split(': ')[1])
        
        
        '''assert 'Train_set:' in values[22]
        train_valid.append(float(values[23].split(': ')[-1].strip('%\n')))
        train_same.append(float(values[24].split(': ')[-1].strip('\n')))
        
        assert 'Validation_set:' in values[26]
        validation_valid.append(float(values[27].split(': ')[-1].strip('%\n')))
        validation_same.append(float(values[28].split(': ')[-1].strip('\n')))
        
        assert 'Test_set:' in values[30]
        test_valid.append(float(values[31].split(': ')[-1].strip('%\n')))
        test_same.append(float(values[32].split(': ')[-1].strip('\n')))'''
        
        assert 'Valid' in values[21]
        test_valid.append(float(values[21].split(': ')[-1].strip('%\n')))
        
        validation_loss_total.append(val_loss)
        validation_loss_recon.append(val_recon_loss)
        validation_loss_kl.append(val_kl_loss)
        
        train_loss_total.append(training_loss)
        train_loss_recon.append(train_recon_loss)
        train_loss_kl.append(train_kl_loss)
        
        test_loss_total.append(test_loss)
        test_loss_recon.append(test_recon_loss)
        test_loss_kl.append(test_kl_loss)
        
        loss_params.append(file.strip('\n'))
        
    except:
        check.append(file.strip('\n'))
    
with open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/check_cases.txt', 'w') as f:
    f.write('\n'.join(check))

print(validation_loss_total)

d = pd.DataFrame(validation_loss_total, columns = ['Val_loss_total'])
d['Val_loss_recon'] = validation_loss_recon
d['Val_loss_kl'] = validation_loss_kl
# d['Val_valid'] = validation_valid
# d['Val_same'] = validation_same
d['Train_loss_total'] = train_loss_total
d['Train_loss_recon'] = train_loss_recon
d['Train_loss_kl'] = train_loss_kl
# d['Train_valid'] = train_valid
# d['Train_same'] = train_same
d['Test_loss_total'] = test_loss_total
d['Test_loss_recon'] = test_loss_recon
d['Test_loss_kl'] = test_loss_kl
# d['Test_valid'] = test_valid
# d['Test_same'] = test_same
d['Valid'] = test_valid
d['Difference'] = np.abs(d['Train_loss_total'] - d['Val_loss_total'])
d['Parameters'] = loss_params
# d = d.sort_values(['Val_loss_total'])
d = d.sort_values(['Valid'], ascending = False)
print(d.shape)
d.to_csv('summary_results_old.csv', header=True, index=False)

best_parameters = d.head(20)
# best_parameters.to_csv("/hps/research1/icortes/acunha/python_scripts/Molecular_vae//best_parameters_pancancer_losses.txt")
# best_parameters = best_parameters.sort_values('Difference')
print(best_parameters.head(20))
print('\n'.join(list(best_parameters['Parameters'].head(20))))
'''
new_file = "_".join(f.split("_")[-2:])
with open(/hps/research1/icortes/acunha/python_scripts/Molecular_vae/list_best_parameters_{}.txt'.format(new_file), 'w') as f:
    f.write(list(best_parameters['Parameters'].head(1))[0])
    f.write('\n')
    f.write(list(best_parameters['Parameters'].head(1))[0])'''
