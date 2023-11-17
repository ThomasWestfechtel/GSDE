task=('A2C' 'A2P' 'A2R' 'C2A' 'C2P' 'C2R' 'P2A' 'P2C' 'P2R' 'R2A' 'R2C' 'R2P')
source=('Art' 'Art' 'Art' 'Clipart' 'Clipart' 'Clipart' 'Product' 'Product' 'Product' 'RealWorld' 'RealWorld' 'RealWorld')
target=('Clipart' 'Product' 'RealWorld' 'Art' 'Product' 'RealWorld' 'Art' 'Clipart' 'RealWorld' 'Art' 'Clipart' 'Product')

method='CDAN'
net='ResNet50'
dset='office-home'
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
    s_dset_path='../OfficeHome/'${source[index]}'/label_c.txt'
    t_dset_path='../OfficeHome/'${target[index]}'/label_c.txt'
    output_dir='OH_'${mm_ver}'_'${mb_ver}'_'${pl_ver}'/'${seed[0]}'/'${task[index]}
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
