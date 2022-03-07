export CUDA_VISIBLE_DEVICES='0,2'

stage=2
task="se" #"vocoder" or "se"
model_name="voicebank_se_mixc"
# pretrain_model="voicebank_model_vocoder/weights-787725.pt"
# fix="--fix_in"
. ./path.sh

voicebank_noisy="${voicebank}/noisy_trainset_28spk_wav_16k"
voicebank_clean="${voicebank}/clean_trainset_28spk_wav_16k"


if [[ ! " se vocoder " =~ " $task " ]]; then
  echo "Error: \$task must be either se or vocoder: ${task}"
  exit 1;
fi


if [[ "$task" == "se" ]]; then
    wav_root=${voicebank_noisy}
    spec_root=${diffwave}/spec/voicebank_Noisy
    spec_type="noisy spectrum"

elif [[ "$task" == "vocoder" ]]; then
    wav_root=${voicebank_clean}
    spec_root=${diffwave}/spec/voicebank_Clean
    spec_type="clean Mel-spectrum"
fi

if [ ${stage} -le 1 ]; then
    echo "stage 1 : preparing training data"
    wave_path=${wav_root}
    echo "create ${spec_type} from ${wave_path} to ${spec_root}"
    rm -r ${spec_root} 2>/dev/null
    mkdir -p ${spec_root}
    python src/diffwave/preprocess.py ${wave_path} ${spec_root} --${task} --voicebank
    mkdir -p ${spec_root}/train
    mkdir -p ${spec_root}/valid
    mv ${spec_root}/p226_*.wav.spec.npy ${spec_root}/valid
    mv ${spec_root}/p287_*.wav.spec.npy ${spec_root}/valid
    mv ${spec_root}/*.wav.spec.npy ${spec_root}/train

fi

if [ ${stage} -le 2 ]; then
    echo "stage 2 : training model"
    target_wav_root=${voicebank_clean}
    noisy_wav_root=${voicebank_noisy}

    train_spec_list=""

    spec_path=${spec_root}/train
    train_spec_list="${train_spec_list} ${spec_path}"
    
    if [ -z "$pretrain_model" ]; then
        python src/diffwave/__main__.py ${diffwave}/${model_name} ${target_wav_root} ${noisy_wav_root} ${train_spec_list}  --${task} ${fix}  --voicebank
    else
        python src/diffwave/__main__.py ${diffwave}/${model_name} ${target_wav_root} ${noisy_wav_root} ${train_spec_list}  --${task} --pretrain ${diffwave}/${pretrain_model} ${fix}  --voicebank
    fi
fi

