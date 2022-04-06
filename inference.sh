stage=$1
ckp=$2
task=$3 #"se_pre" or "se"
model_name=$4

. ./path.sh

voicebank_noisy="${voicebank}/noisy_testset_wav"
voicebank_clean="${voicebank}/clean_testset_wav"


if [[ ! " se se_pre " =~ " $task " ]]; then
  echo "Error: \$task must be either se or se_pre: ${task}"
  exit 1;
fi


if [[ "$task" == "se" ]]; then
    wav_root=${voicebank_noisy}
    spec_root=${output_path}/spec/voicebank_Noisy_Test
    spec_type="noisy spectrum"

elif [[ "$task" == "se_pre" ]]; then
    wav_root=${voicebank_clean}
    spec_root=${output_path}/spec/voicebank_Clean_Test
    spec_type="clean Mel-spectrum"
fi


if [ ${stage} -le 1 ]; then
    echo "stage 1 : preparing testing data"

    spec_path=${spec_root}
    wave_path=${wav_root}
    echo "create ${spec_type} from ${wave_path} to ${spec_path}"
    rm -r ${spec_path} 2>/dev/null
    mkdir -p ${spec_path}
    python src/cdiffuse/preprocess.py ${wave_path} ${spec_path} --${task} --test --voicebank
fi

if [ ${stage} -le 2 ]; then
    echo "stage 2 : inference model"

    test_spec_list=${spec_root}
    
    enhanced_path=${output_path}/Enhanced/${model_name}/model${ckp}/test
    rm -r ${enhanced_path} 2>/dev/null
    mkdir -p ${enhanced_path} 
    echo "inference enhanced wav file from ${spec_root} to ${enhanced_path}"
    python src/cdiffuse/inference.py  ${output_path}/${model_name}/weights-${ckp}.pt ${test_spec_list} ${voicebank_noisy} -o ${enhanced_path} --${task} --voicebank
fi

# if [ ${stage} -le 3 ]; then
#     echo "stage 3 : scoring"
#     score_file=${output_path}/Enhanced/${model_name}/scores.csv
#     clean_wav=${voicebank_clean}
#     enhanced_result=${output_path}/Enhanced/${model_name}/model${ckp}/test
#     echo "save the score to ${score_file}"
#     # cd pesq/speech-metrics/
#     # python main.py ${clean_wav} ${enhanced_result} ${score_file} model${ckp}
#     # cd -
# fi