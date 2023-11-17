task=('R2C' 'R2S' 'R2P' 'C2R' 'C2S' 'C2P' 'S2R' 'S2C' 'S2P' 'P2R' 'P2C' 'P2S')
source=('real' 'real' 'real' 'clipart' 'clipart' 'clipart' 'sketch' 'sketch' 'sketch' 'painting' 'painting' 'painting')
target=('clipart' 'sketch' 'painting' 'real' 'sketch' 'painting' 'real' 'clipart' 'painting' 'real' 'clipart' 'sketch')

method='CDAN'
net='ResNet50'
dset='domain-net'
seed=('42' '43' '44')
index=0
t_inter='5000'
num_iter='5004'
mm_ver='1'
mb_ver='1'
pl_ver='1'
perc=('0' '0.2' '0.4' '0.6' '0.8' '0.9')

echo $index

for((run_id=0; run_id < 5; run_id++))
  do	  
    echo ">> Seed 42: traning task ${index} : ${task[index]}"
    s_dset_path='../DomainNet/'${source[index]}'_train_mini.txt'
    t_dset_path='../DomainNet/'${target[index]}'_train_mini.txt'
    output_dir='DN_'${mm_ver}'_'${mb_ver}'_'${pl_ver}'/'${seed[0]}'/'${task[index]}
    python main.py \
       ${method} \
       --net ${net} \
       --dset ${dset} \
       --s_dset_path ${s_dset_path} \
       --t_dset_path ${t_dset_path} \
       --output_dir ${output_dir} \
       --test_interval ${t_inter} \
       --num_iter ${num_iter} \
       --run_id ${run_id} \
       --perc ${perc[run_id]} \
       --mm_ver ${mm_ver} \
       --mb_ver ${mb_ver} \
       --pl_ver ${pl_ver} \
       --seed ${seed[0]} 
done
