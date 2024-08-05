
# State-of-the-art Wav2vec2.0 finetuning approach
- raw audio (wihtout sliding window. 16khz)
- Pretrained wav2vec encoder (which includes a CNN and a transformer)
- Linear layer on top of encoder representations (targets LM dictionary size # of classes)
- CTC loss on the output of linear layer


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
    - fontend: SlidingWindow, window_size 1
    - encoder: wav2vec2=FairSeqWav2Vec2Encoder,
    - CTC layer (linear + CTC loss)
        - issue Required FairSeqWav2Vec2Encoder, which depends on omegaconf version that is incompatible with s3prl
            - could downgrade omegaconf (assuming no other part of the system depends on omegaconf)
