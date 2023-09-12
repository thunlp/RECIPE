# Removing Backdoors in Pre-trained Models by Regularized Continual Pre-training

(1) Our defense code is in the RECIPE.py file.

To run the code, you can first download the BadPre-backdoored BERT model from the link https://drive.google.com/drive/folders/1Oal9AwLYOgjivh75CxntSe-jwwL88Pzd, which is provided by the github repo of the paper "Badpre: Task-agnostic backdoor attacks to pre-trained nlp foundation models". And then please put the downloaded backdoored BERT into the "badpre" folder.
    
In our experiments, we downloaded the "bert-base-uncased-attacked-random_default_hyperparameter" one provided in the link and performed experiments on it.
    
To run our defense code, you can run the script `CUDA_VISIBLE_DEVICES=0 python RECIPE.py`, then you can get a purified model.
    
(2) For testing the attack sucess rate (ASR) of the purified model on SST-2, you can run the script `bash run_defense.sh`.

(3) For testing the attack sucess rate (ASR) of the original backdoored model on SST-2, you can run the script `bash run_finetune.sh`.

(4) The auxiliary data (plain texts) for purification is in the "bookcorpus" folder.

(5) The downstream SST-2 data is in the "data" folder.
