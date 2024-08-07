
# State-of-the-art Wav2vec2.0 finetuning approach
- raw audio (wihtout sliding window. 16khz)
- Pretrained wav2vec encoder (which includes a CNN and a transformer)
- Linear layer on top of encoder representations (targets LM dictionary size # of classes)
- CTC loss on the output of linear layer



[frontend, pre-encoder, encoder, decoder,  (linear, loss)]


[s3plr, preenc, 2ltransformers, none, (linear, loss)]

[s3plr, none, none, none, (linear, loss)]

[none, none, fairseqwav2vec2, none, (linear, loss)]


# TODO
- [ ] Externalize language names:
    - see: /nlp/scr/moussa/git/asr-dro/espnet/egs2/asr_dro/asr1/local/data_prep.py
- [ ] decide how to use wav2vec frontend
- [ ] dedide how to use whispers
- [ ] finalize configs for baselines and DRO
- [ ] run ASR tuning experiments for 143 languages 
 

Option 1:
    - fontend: s3prl wav2vec xlsr300/mms
    - encoder: Custom Identity (does nothing)
    - CTC layer (linear + CTC loss)



Option 2:
    - fontend: NONE
    - encoder: wav2vec2=FairSeqWav2Vec2Encoder,
    - CTC layer (linear + CTC loss)
        - issue Required FairSeqWav2Vec2Encoder, which depends on omegaconf version that is incompatible with s3prl
            - could downgrade omegaconf (assuming no other part of the system depends on omegaconf)




# Current issue
1): omegaconf dependency hell ()
2): fairseq apparently inapropriately configured by espnet adapter
    - see: /juice3/scr3/moussa/git/asr-dro/espnet/espnet2/asr/encoder/wav2vec2_encoder.py line 60
[jagupard29] 2024-06-21 15:53:49,226 (wav2vec2_encoder:166) INFO: Wav2Vec model ./downloads/wav2vec_pretrained_models/xlsr2_300m.pt already exists.
Traceback (most recent call last):
  File "/nlp/scr/moussa/git/asr-dro/espnet/tools/miniconda/envs/espnet/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/nlp/scr/moussa/git/asr-dro/espnet/tools/miniconda/envs/espnet/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/juice3/scr3/moussa/git/asr-dro/espnet/espnet2/bin/asr_train.py", line 23, in <module>
    main()
  File "/juice3/scr3/moussa/git/asr-dro/espnet/espnet2/bin/asr_train.py", line 19, in main
    ASRTask.main(cmd=cmd)
  File "/juice3/scr3/moussa/git/asr-dro/espnet/espnet2/tasks/abs_task.py", line 1157, in main
    cls.main_worker(args)
  File "/juice3/scr3/moussa/git/asr-dro/espnet/espnet2/tasks/abs_task.py", line 1267, in main_worker
    model = cls.build_model(args=args)
  File "/juice3/scr3/moussa/git/asr-dro/espnet/espnet2/tasks/asr.py", line 561, in build_model
    encoder = encoder_class(input_size=input_size, **args.encoder_conf)
  File "/juice3/scr3/moussa/git/asr-dro/espnet/espnet2/asr/encoder/wav2vec2_encoder.py", line 60, in __init__
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
  File "/nlp/scr/moussa/git/asr-dro/espnet/tools/miniconda/envs/espnet/lib/python3.8/site-packages/fairseq/checkpoint_utils.py", line 436, in load_model_ensemble_and_task
    task = tasks.setup_task(cfg.task)
  File "/nlp/scr/moussa/git/asr-dro/espnet/tools/miniconda/envs/espnet/lib/python3.8/site-packages/fairseq/tasks/__init__.py", line 39, in setup_task
    cfg = merge_with_parent(dc(), cfg)
  File "/nlp/scr/moussa/git/asr-dro/espnet/tools/miniconda/envs/espnet/lib/python3.8/site-packages/fairseq/dataclass/utils.py", line 500, in merge_with_parent
    merged_cfg = OmegaConf.merge(dc, cfg)
  File "/nlp/scr/moussa/git/asr-dro/espnet/tools/miniconda/envs/espnet/lib/python3.8/site-packages/omegaconf/omegaconf.py", line 321, in merge
    target.merge_with(*others[1:])
  File "/nlp/scr/moussa/git/asr-dro/espnet/tools/miniconda/envs/espnet/lib/python3.8/site-packages/omegaconf/basecontainer.py", line 331, in merge_with
    self._format_and_raise(key=None, value=None, cause=e)
  File "/nlp/scr/moussa/git/asr-dro/espnet/tools/miniconda/envs/espnet/lib/python3.8/site-packages/omegaconf/base.py", line 95, in _format_and_raise
    format_and_raise(
  File "/nlp/scr/moussa/git/asr-dro/espnet/tools/miniconda/envs/espnet/lib/python3.8/site-packages/omegaconf/_utils.py", line 629, in format_and_raise
    _raise(ex, cause)
  File "/nlp/scr/moussa/git/asr-dro/espnet/tools/miniconda/envs/espnet/lib/python3.8/site-packages/omegaconf/_utils.py", line 610, in _raise
    raise ex  # set end OC_CAUSE=1 for full backtrace
  File "/nlp/scr/moussa/git/asr-dro/espnet/tools/miniconda/envs/espnet/lib/python3.8/site-packages/omegaconf/basecontainer.py", line 329, in merge_with
    self._merge_with(*others)
  File "/nlp/scr/moussa/git/asr-dro/espnet/tools/miniconda/envs/espnet/lib/python3.8/site-packages/omegaconf/basecontainer.py", line 347, in _merge_with
    BaseContainer._map_merge(self, other)
  File "/nlp/scr/moussa/git/asr-dro/espnet/tools/miniconda/envs/espnet/lib/python3.8/site-packages/omegaconf/basecontainer.py", line 314, in _map_merge
    dest[key] = src._get_node(key)
  File "/nlp/scr/moussa/git/asr-dro/espnet/tools/miniconda/envs/espnet/lib/python3.8/site-packages/omegaconf/dictconfig.py", line 258, in __setitem__
    self._format_and_raise(
  File "/nlp/scr/moussa/git/asr-dro/espnet/tools/miniconda/envs/espnet/lib/python3.8/site-packages/omegaconf/base.py", line 95, in _format_and_raise
    format_and_raise(
  File "/nlp/scr/moussa/git/asr-dro/espnet/tools/miniconda/envs/espnet/lib/python3.8/site-packages/omegaconf/_utils.py", line 629, in format_and_raise
    _raise(ex, cause)
  File "/nlp/scr/moussa/git/asr-dro/espnet/tools/miniconda/envs/espnet/lib/python3.8/site-packages/omegaconf/_utils.py", line 610, in _raise
    raise ex  # set end OC_CAUSE=1 for full backtrace
omegaconf.errors.ConfigKeyError: Key 'multiple_train_files' not in 'AudioPretrainingConfig'
	full_key: multiple_train_files
	reference_type=Optional[AudioPretrainingConfig]
	object_type=AudioPretrainingConfig
# Accounting: time=51 threads=1




# 2024-06-24
# Encoder/CTC
- Overall
  - Use tri-stage LR scheduler.
- XLSR: [CNN, Transformer Encoder, Linear+CTC]
  - freeze CNN for finetuning, train the rest
  - (under-specified by paper. Martijn found this in fairserq code)
    - https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/xlsr/config/finetune.yaml
    - see: feature_grad_mult
- MMS: [CNN, Transformer Encoder, Linear+CTC]
  - full finetuning
# Encoder/Decoder
- Whispers

 freeze_finetune_updates: 10000




 # 20240805
 - Notes on Whispers / Huggingface
 - Main components of interest: Trainer and TrainerArguments
  - These are also specialixzed as Seq2seqTrainer and seq2secTrainerArgs
  - Also need to look at data batching (e.g. audio length equalization across languages)
  - Also need to look at data loader
    - ensure that language ID is available in batch fields
  - Also need to check that the Trainer is stateful (because we need need to maintain q vectors in a persistent way throughout the training session)
- Notes on Whispers / ESPNET
- There are some whisper components in ESPNEt (including, an encoder, a decoder, and examples of finetuning experiments.)
- However it was not exacly clear how ESPNET was configured to run the cross entropy loss.