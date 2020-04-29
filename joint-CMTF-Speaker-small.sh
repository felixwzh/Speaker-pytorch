#!/bin/bash

# in this bash file, we train the speaker model and S3CMTF together

##### params that we want to use #####

# general param
# 1. total number of training rounds. each round means several rounds of speaker and several rounds of S3CMTF
total_rounds_num=4
# 2. rounds that we train speaker model
speaker_rounds_num=2
# 3. rounds that we train S3CMTF model
S3CMTF_rounds_num=2
exp_prefix="joint_exp_one"
log_file="/home/felix/project/Speaker-pytorch/log/${exp_prefix}_general.log"


# S3CMTF param
S3CMTF_base_dir="/home/felix/project/PerGen/src_data_NEW_small_PCA_dim_384"
S3CMTF_prefix=${exp_prefix}
S3CMTF_last_rounds_dir='results_lr_5e-4_1_4_15x30x20' #todo
S3CMTF_lr=0.0005
S3CMTF_lambda_reg=1
S3CMTF_lambda_c=4

# spkear param
speaker_base_dir="/home/felix/project/Speaker-pytorch"
speaker_data_folder="data/src_data_NEW_small_0428"
speaker_save_prefix="speaker.model"
speaker_batch_size=200
speaker_layers=2
speaker_epochs=${speaker_rounds_num}
speaker_eval_steps=500
speaker_PersonaNum=219
speaker_save_steps=1
speaker_train_size=657
speaker_PersonaDim=30 
seapker_fine_tunine_model="None"
speaker_save_folder="${speaker_base_dir}/save/${exp_prefix}/round_${round}"
seapker_persona_emb_output_file="${speaker_save_folder}/${speaker_save_prefix}.user.emb"
speaker_PersonaEmbFiles="${S3CMTF_base_dir}/${S3CMTF_last_rounds_dir}/FACTOR2_iter_${S3CMTF_rounds_num}"

##### start the loop #####
echo "joint training start" 2>&1 | tee -a ${log_file}
round=1
while(( ${round}<=${total_rounds_num} ))
do



echo "start round ${round}"  2>&1 | tee -a ${log_file}

echo "at ${speaker_base_dir}"  2>&1 | tee -a ${log_file}
# todo: log the main loop to a file. while also log everything to another file. 
# step 1. run speaker at ~/project/Speaker-pytorch
cd ${speaker_base_dir}


echo "at ${speaker_base_dir}"  2>&1 | tee -a ${log_file}

# param update
speaker_save_folder="${speaker_base_dir}/save/${exp_prefix}/round_${round}"
seapker_persona_emb_output_file="${speaker_save_folder}/${speaker_save_prefix}.user.emb"
speaker_PersonaEmbFiles="${S3CMTF_base_dir}/${S3CMTF_last_rounds_dir}/FACTOR2_iter_${S3CMTF_rounds_num}"

# python train.py \
#     --data_folder ${speaker_data_folder} \
#     --save_folder  ${speaker_save_folder} \
#     --save_prefix  ${speaker_save_prefix} \
#     --output_file "${speaker_save_prefix}.log" \ 
#     --SpeakerMode --PersonaNum ${speaker_PersonaNum} \
#     --PersonaEmbFiles ${speaker_PersonaEmbFiles} \
#     --batch_size ${speaker_batch_size} --layers ${speaker_layers} \
#     --epochs ${speaker_epochs} --eval_steps ${speaker_eval_steps} \
#     --save_steps ${speaker_save_steps} --train_size ${speaker_train_size} \
#     --PersonaDim ${speaker_PersonaDim} \
#     --fine_tuning --fine_tunine_model ${seapker_fine_tunine_model} \
#     --output_persona_emb_in_training --persona_emb_output_file ${seapker_persona_emb_output_file}
python train.py  --data_folder ${speaker_data_folder} --save_folder  ${speaker_save_folder} --save_prefix  ${speaker_save_prefix} --output_file "${speaker_save_prefix}.log" \--SpeakerMode --PersonaNum ${speaker_PersonaNum} --PersonaEmbFiles ${speaker_PersonaEmbFiles} --batch_size ${speaker_batch_size} --layers ${speaker_layers} --epochs ${speaker_epochs} --eval_steps ${speaker_eval_steps} --save_steps ${speaker_save_steps} --train_size ${speaker_train_size} --PersonaDim ${speaker_PersonaDim} --fine_tuning --fine_tunine_model ${seapker_fine_tunine_model} --output_persona_emb_in_training --persona_emb_output_file ${seapker_persona_emb_output_file}

# 1.2 post process some var for next training. 
seapker_fine_tunine_model="${speaker_save_folder}/${speaker_save_prefix}${speaker_rounds_num}"

echo "finished spkear"
# step 2. save the user embs file to the right place
cd ${S3CMTF_base_dir}
echo "at ${S3CMTF_base_dir}"
# 2.1 build a folder to contain all the FACTORS that we will use
# NOTE that we will user all the pretrained FACTORS because in this way we can have more stable user embeddings. isInputPath=1
# TODO: try other ways as well, e.g., isInputPath=2 
S3CMTF_init_factors_dir="${S3CMTF_prefix}_round_${round}_init_factors"
mkdir ${S3CMTF_init_factors_dir}

# 2.2 cp persona_emb
cp ${seapker_persona_emb_output_file} "${S3CMTF_init_factors_dir}/FACTOR2"

# 2.3 cp other files
cp "${S3CMTF_last_rounds_dir}/FACTOR1_iter_${S3CMTF_rounds_num}" "${S3CMTF_init_factors_dir}/FACTOR1"
# cp "${S3CMTF_last_rounds_dir}/FACTOR2_iter_${S3CMTF_rounds_num}" "${S3CMTF_init_factors_dir}/FACTOR2"
cp "${S3CMTF_last_rounds_dir}/FACTOR3_iter_${S3CMTF_rounds_num}" "${S3CMTF_init_factors_dir}/FACTOR3"
cp "${S3CMTF_last_rounds_dir}/CFACTOR1_iter_${S3CMTF_rounds_num}" "${S3CMTF_init_factors_dir}/CFACTOR1"
cp "${S3CMTF_last_rounds_dir}/CFACTOR2_iter_${S3CMTF_rounds_num}" "${S3CMTF_init_factors_dir}/CFACTOR2"
cp "${S3CMTF_last_rounds_dir}/CORETENSOR_iter_${S3CMTF_rounds_num}" "${S3CMTF_init_factors_dir}/CORETENSOR"

# step 3. run S3CMTF 
S3CMTF_cur_rounds_output_dir="${S3CMTF_prefix}_round_${round}_output"

./S3CMTF-opt-con s3cmtf_config.txt ${S3CMTF_init_factors_dir} s3cmtf_dev.tensor ${S3CMTF_cur_rounds_output_dir} 2 ${S3CMTF_lr} ${S3CMTF_lambda_reg} ${S3CMTF_lambda_c} ${S3CMTF_rounds_num} 0.1 2 0 0.1 0 1

echo "finished s3cmtf"

S3CMTF_last_rounds_dir=$S3CMTF_cur_rounds_output_dir

let "round++"
done