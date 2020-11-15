with open('check_cases.txt', 'r') as f:
    lines = f.readlines()
    
lines = [x.strip('\n') for x in lines]

for line in lines:
    line = line.split('/')
    parameters = line[3].strip('output_').strip('.txt').split('_')
    if parameters[-3] == 'non':
        new_p = parameters[:-3]
        new_p.append('{}_{}'.format(parameters[-3], parameters[-2]))
        new_p.append(parameters[-1])
        new = '{}-{}-{}'.format(line[1].split('_')[1], line[2], '-'.join(new_p))
    else:
        new = '{}-{}-{}'.format(line[1].split('_')[1], line[2], '-'.join(parameters))
    with open('ready_check_cases.txt', 'a') as f:
        f.write(new)
        f.write('\n')
