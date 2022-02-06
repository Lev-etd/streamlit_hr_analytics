import os
#
# asr = ASREngine("ru", model_path="jonatasgrosman/wav2vec2-large-xlsr-53-russian", device='cuda:0', inference_batch_size=1)
#
# transcriptions = asr.transcribe(audio_paths)
# speller = YandexSpeller(lang='ru')
# fixed = speller.spelled(transcriptions[0]['transcription'])

if __name__ == '__main__':
    print()
    # print(os.listdir('./uploaded_videos'))