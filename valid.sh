../../valid_vb.shstage=$1
ckp=$2
task="vocoder" #"vocoder" or "se"
model_name="voicebank_model_vocoder"

. ./path.sh

voicebank_noisy="${voicebank}/noisy_trainset_28spk_wav_16k"
voicebank_clean="${voicebank}/clean_trainset_28spk_wav_16k"


if [[ ! " se vocoder " =~ " $task " ]]; then
  echo "Error: \$task must be either se or vocoder: ${task}"
  exit 1;
fi


if [[ "$task" == "se" ]]; then
    wav_root=${voicebank_noisy}
    spec_root=${diffwave}/spec/voicebank_Noisy/valid
    spec_type="noisy spectrum"

elif [[ "$task" == "vocoder" ]]; then
    wav_root=${voicebank_clean}
    spec_root=${diffwave}/spec/voicebank_Clean/valid
    spec_type="clean Mel-spectrum"
fi


# if [ ${stage} -le 1 ]; then
#     echo "stage 1 : preparing testing data"

#     spec_path=${spec_root}
#     wave_path=${wav_root}
#     echo "create ${spec_type} from ${wave_path} to ${spec_path}"
#     rm -r ${spec_path} 2>/dev/null
#     mkdir -p ${spec_path}
#     python src/diffwave/preprocess.py ${wave_path} ${spec_path} --${task} --test --voicebank
# fi

if [ ${stage} -le 2 ]; then
    echo "stage 2 : inference model"
    target_wav_root=${voicebank_clean}

    test_spec_list=${spec_root}
    
    enhanced_path=${diffwave}/Enhanced/${model_name}/model${ckp}/
    rm -r ${enhanced_path} 2>/dev/null
    mkdir -p ${enhanced_path} 
    echo "inference enhanced wav file from ${spec_root} to ${enhanced_path}"
    
    python src/diffwave/inference.py  ${diffwave}/${model_name}/weights-${ckp}.pt ${test_spec_list} ${voicebank_noisy} -o ${enhanced_path} --${task} --voicebank
fi

if [ ${stage} -le 3 ]; then
    echo "stage 3 : scoring"
    score_file=${diffwave}/Enhanced/${model_name}/scores.csv
    clean_wav=${diffwave}/clean_dev/
    enhanced_result=${diffwave}/Enhanced/${model_name}/model${ckp}/valid
    echo "save the score to ${score_file}"
    cd pesq/speech-metrics/
    python main.py ${clean_wav} ${enhanced_result} ${score_file} model${ckp}
    cd -
fi