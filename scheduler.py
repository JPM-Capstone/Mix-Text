import os

def main():

    ### Setup Directories

    base_dir = './mixtext_jobs/'

    # Make base Log dir
    logs_base_dir = os.path.join(base_dir,'logs')
    os.makedirs(logs_base_dir,exist_ok=True)

    # Make logs dir for next batch
    batch_number = max([int(d.split('_')[-1]) for d in os.listdir(logs_base_dir)]+[1])
    logs_dir = os.path.join(logs_base_dir,f'batch_{batch_number}')
    os.makedirs(logs_dir,exist_ok=True)

    if len(os.listdir(logs_dir))!=0:
        logs_dir = os.path.join(logs_base_dir,f'batch_{batch_number+1}')

    os.makedirs(logs_dir,exist_ok=True)

    # Make scripts dir
    scripts_dir = os.path.join(base_dir,'scripts')
    os.makedirs(scripts_dir,exist_ok=True)

    # Make jobs dir
    jobs_dir = os.path.join(base_dir,'sbatch')
    os.makedirs(jobs_dir,exist_ok=True)

    AS_logs_dir = "/home/as14229/NYU_HPC/Mix-Text" + logs_dir[1:]

    ### Setup Headers

    sbatch_header = f"#!/bin/bash\n\
\n\
#SBATCH --nodes=1               \n\
#SBATCH --ntasks-per-node=1     \n\
#SBATCH --gres=gpu:1            \n"

    job_params = dict(     
        time = '24:00:00',                # Time per job
        memory = '32GB',                       # RAM required in GB
        partition = 'a100_1,a100_2,rtx8000')    # GPUs you want, to list all available run - partition list - sinfo -s

    sbatch_header+=f'#SBATCH --partition={job_params["partition"]}\n'
    sbatch_header+=f'#SBATCH --cpus-per-task=4\n'
    sbatch_header+=f'#SBATCH --mem={job_params["memory"]}\n'
    sbatch_header+=f'#SBATCH --time={job_params["time"]}\n'

    job_name_directive =  "#SBATCH --job-name=Job"
    output_file_directive = "#SBATCH --output="+AS_logs_dir+'/job'

    # Command Header
    command_header = "\n\
source ~/.bashrc\n\
conda activate uda\n\
cd /home/as14229/NYU_HPC/Mix-Text/\n\n"

    # Main Commmand
    command = "python train.py "

    ### Get all Full Commands

    num_labeled = [100, 500, 1000, 5000]
    idrs = [5, 25, 50, 75, 100]

    commands = []
    for num in num_labeled:
        for idr in idrs:
            commands.append(command + f"labeled_{num}_idr_{idr}")
            

    ### Make SBATCH Files

    # Get the Next Job number
    try: job_start_number = sorted([int(m.split('.')[0][3:]) for m in os.listdir(jobs_dir)])[-1]+1
    except: job_start_number = 1

    # Number of consecutive jobs per GPU
    jobs_per_gpu = 1

    # Make sbatch files
    for i,_ in enumerate(range(0,len(commands),jobs_per_gpu),job_start_number):
        with open(os.path.join(jobs_dir,'job'+str(i)+'.sbatch'),'w') as file:
            file.write(sbatch_header)
            file.write(job_name_directive+str(i)+'\n')
            file.write(output_file_directive+str(i)+'.log\n')
            file.write(command_header)
            file.write(commands[i - 1])

    ### Make Schedule File

    to = sorted([int(m.split('.')[0][3:]) for m in os.listdir(jobs_dir)])[-1]
    from_ = sorted([int(m.split('.')[0][3:]) for m in os.listdir(jobs_dir)])[0]

    schedule_file = os.path.join(scripts_dir,'schedule_jobs.sh')
    with open(schedule_file,'w') as file:
        file.write('#!/bin/bash\n\n')
        for k in range(from_,to+1):
            file.write('sbatch '+jobs_dir+'/job'+str(k)+'.sbatch\n')
    os.chmod(schedule_file, 0o740)

if __name__ == '__main__':
    main()
