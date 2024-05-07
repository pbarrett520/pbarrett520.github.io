---
title: "Using VITS on Coqui TTS to Train and Compare Two Esperanto Models"
excerpt: <br/><img src='https://pbarrett520.github.io/images/tts-image.jpg'>
collection: portfolio
---

The basic idea behind this project is very simple. I wanted to see what would happen if I stripped all of the punctuation out of the training data for a text-to-speech model. I was curious what effect (if any) this might have on a TTS model's performance. To pursue this line of inquiry, I trained two Esperanto text-to-speech (TTS) models using the 6/3/2019 version of the Common Voice Esperanto dataset. The dataset contains 446.9 MB of data, including 14 hours of validated recordings. Why Esperanto, one may ask? My answer is, "Why not?". Esperanto is fun and conlangs have always been an interest of mine. The training parameters for each model were identical, and are shown below:

```python
#here are the Esperanto characters
characters = CharactersConfig(
    characters="KoNtndOzŜheĉĴmsuPEĜŝIDZaĈjJSŭFlkTĥULgMHvĤpĝbRBGrCifĵcVA",
    punctuations=".',\"(—! );-?:",
    characters_class="TTS.tts.models.vits.VitsCharacters",
    pad="<PAD>",
)

dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="metadata.csv",
    path=os.path.join(output_path,"/home/u6/pbarrett520/vits")
)

audio_config = BaseAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    preemphasis=0.0,
    ref_level_db=20,
    log_func="np.log",
    do_trim_silence=True,
    trim_db=45,
    mel_fmin=0,
    mel_fmax=None,
    spec_gain=1.0,
    signal_norm=False,
    do_amp_to_db_linear=False,
)

config = VitsConfig(
    characters=characters,
    audio=audio_config,
    run_name="vits_esperanto",
    batch_size=16,
    eval_batch_size=16,
    batch_group_size=5,
    num_loader_workers=0,
    num_eval_loader_workers=2,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=300,
    text_cleaner="basic_cleaners",
    use_phonemes=False,
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=True,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    test_sentences=[
        ["Mi sentas la mankon de vi."],
        ["Kiel vi povas fari tion al mi?"],
        ["Mi ŝatasrenkonti novajn homojn."]
    ],
)
```

## Pruning Punctuation

Where these two models differ is in the metadata files used to annotate the audio recordings. One file, `no_punct.csv`, has had all punctuation characters stripped out of it using a simple Python script called `no_punct.py`, shown here:

```python
input_file_path = "metadata.csv"
output_file_path = '/Users/patrick/Documents/speech_tech_final/no_punct.csv'

chars_to_remove = r".',\"(—!);-?:" # target characters to be removed

with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
    
    for line in infile:
        for char in chars_to_remove:
            line = line.replace(char, '')

        outfile.write(line)
```

## Training and Using the HPC

After this, each model was invoked to synthesize speech using the Coqui TTS toolkit with the same snippet of text, the first few lines of the Bible in Esperanto. I felt this input was especially useful since it contained lots of punctuation, which is what we're trying to test. The resulting WAV files from the punctuated and unpunctuated versions of the model are named `esperanto_final.wav` and `esperanto_no_punct.wav`, respectively. The training statistics and quality of the outputs were both basically the same. Here is an example of the training stats:

```
--> TIME: 2024-02-25 01:58:59 -- STEP: 652/652 -- GLOBAL_STEP: 191450[0m
     | > loss_disc: 2.7056007385253906  (2.522395754177309)
     | > loss_disc_real_0: 0.15962564945220947  (0.1640304950223709)
     | > loss_disc_real_1: 0.18271289765834808  (0.21253302690891734)
     | > loss_disc_real_2: 0.22832253575325012  (0.21968462823138285)
     | > loss_disc_real_3: 0.2471582442522049  (0.23464249146242078)
     | > loss_disc_real_4: 0.2348787486553192  (0.23205599518016326)
     | > loss_disc_real_5: 0.2568364441394806  (0.2360907632399635)
     | > loss_0: 2.7056007385253906  (2.522395754177309)
     | > grad_norm_0: tensor(8.1043, device='cuda:0')  (tensor(17.1827, device='cuda:0'))
     | > loss_gen: 2.130958080291748  (2.2033398981110204)
     | > loss_kl: 1.5401383638381958  (1.5755665924065936)
     | > loss_feat: 7.32181978225708  (6.580922028155023)
     | > loss_mel: 19.086673736572266  (18.72859995864157)
     | > loss_duration: 2.8222999572753906  (2.646094435473219)
     | > amp_scaler: 256.0  (256.0)
     | > loss_1: 32.90188980102539  (31.73452292090633)
     | > grad_norm_1: tensor(238.6747, device='cuda:0')  (tensor(201.2404, device='cuda:0'))
     | > current_lr_0: 0.00019302411853756697 
     | > current_lr_1: 0.00019302411853756697 
     | > step_time: 1.5487  (1.4106978479017847)
     | > loader_time: 0.0906  (0.0643565234947838)
```
All training was done on the University of Arizona's high-performance computing cluster. This was a learning experience unto itself. While logging onto the HPC and interacting with it through the Linux command line was straightforward, learning how to submit training tasks to its GPUs via the queue system was not. UAZ's HPC system uses the SLURM Workload Manager to do the scheduling for its resources via an HPC workload container plugin. This means for every Python script you write, you must also create a corresponding SLURM script. You then run your Python script by invoking the relevant SLURM script with the `sbatch` command SLURM provides. After that, you must wait your turn for your script to run, which could take minutes to hours depending on demand. The interface is unwieldy, but I think I got the hang of it after several weeks of practice. Waiting for your code to run is no fun either, but I suppose that is the price you pay for having access to such a powerful tool. I also think this was a really great way to get familiar with accessing remote computing resources more generally. The skills I learned for experimenting with the HPC make a great segway into technologies like AWS and Azure.

## Results and Analysis

The audio of both outputs sounds approximately like human speech, but not quite there yet. Definite breaks between vocalizations can be heard, as well as what seems like some differentiated proto-vowel sounds. Some consonant-like sounds can also be heard. It's interesting that the Esperanto word "cxielo" is audible in `esperanto_final.wav`. There is also a strongly audible lateral approximant /l/ at the end of the same file. In this way, the punctuated model seems to have slightly outperformed. More training epochs and more training data would benefit the quality of both models' outputs tremendously, as neither is intelligible.

However, since punctuation is the main topic of inquiry here, the only rational way to empirically understand what was done is to devise a way to quantitatively measure if the punctuation (or lack thereof) has impacted the model's performance at all. Punctuations often denote some kind of pause in speech. So, measuring the rate of speech in each model's output along with the number of significant pauses in speech seems like a good way to begin some exploratory data analysis. I devised a simple script to quantify some of these things, `pause_analysis.py`:

```python
import wave
import numpy as np

# Function to load WAV file and return numpy array of frames, sample rate, and duration
def load_wav_as_array(file_path):
    with wave.open(file_path, 'r') as wav_file:
        n_frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        duration = n_frames / sample_rate
        audio_data = wav_file.readframes(n_frames)
        # Convert audio bytes to a numpy array based on the sample width
        if wav_file.getsampwidth() == 2:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        elif wav_file.getsampwidth() == 4:
            audio_array = np.frombuffer(audio_data, dtype=np.int32)
        else:
            raise ValueError("Unsupported sample width")
        return (audio_array, sample_rate, duration)

# Function to calculate the speech rate (words per minute)
def calculate_speech_rate(duration, total_words=16):
    return total_words / duration * 60

# Function to find significant pauses in the audio, which might indicate punctuation
def find_significant_pauses(audio, sample_rate, threshold=0.02):
    audio_normalized = audio / np.max(np.abs(audio))
    pauses = np.abs(audio_normalized) < threshold
    pause_lengths = []
    current_pause_length = 0
    for is_pause in pauses:
        if is_pause:
            current_pause_length += 1
        elif current_pause_length > 0:
            pause_lengths.append(current_pause_length / sample_rate)
            current_pause_length = 0
    significant_pauses = [pause for pause in pause_lengths if pause > 0.2]
    return significant_pauses

# File paths
file_path_1 = 'esperanto_final.wav'
file_path_2 = 'esperanto_no_punct.wav'

# Load WAV files and calculate durations
audio_array_1, sample_rate_1, duration_1 = load_wav_as_array(file_path_1)
audio_array_2, sample_rate_2, duration_2 = load_wav_as_array(file_path_2)

# Calculate speech rates using the actual durations
speech_rate_1 = calculate_speech_rate(duration_1)
speech_rate_2 = calculate_speech_rate(duration_2)

# Find significant pauses
significant_pauses_1 = find_significant_pauses(audio_array_1, sample_rate_1)
significant_pauses_2 = find_significant_pauses(audio_array_2, sample_rate_2)

print("Speech Rate 1:", speech_rate_1)
print("Speech Rate 2:", speech_rate_2)
print("Significant Pauses 1:", significant_pauses_1)
print("Significant Pauses 2:", significant_pauses_2)
```

For the first audio file (esperanto_final.wav), its speech rate is about 27.95 words per minute. The speech rates are almost the same, but there is a little bit of an increase in speed from the second one. This might be because punctuation was not used, hence there were fewer pauses. Defined as being longer than 0.2 seconds, there are several significant pauses found in `esperanto_final.wav`. There's an especially long break lasting over 5 seconds, which is likely the break between sentences.

For the second WAV file,`esperanto_no_punct.wav`, the speech rate is approximately 29.32 words per minute. The other WAV file has a large number of breaks, including one longer than five and a half seconds. Despite being generated without punctuation, the presence of pauses implies that TTS system can still pick up on what punctuation *represents*. My hypothesis is that VITS is still able to correlate observed pauses in speech from the audio data with spaces between words in text. In this way, punctuation may be redundant. Perhaps removing punctuation when training a VITS TTS system in the way shown here could result in a marginally more efficient training process. Less characters are less data to be processed, after all. Another experiment would need to be devised to investigate this idea.

## Future Lines of Inquiry

In terms of future investigations, one thing I would definitely like to pursue is optimizing the hyperparameters. I must admit that not a lot of thought was given to the hyperparameters in this experiment, as I was spending a lot of energy figuring out how to use the High-Performance Computing cluster I was running this code one. I think a grid search to find the best combination of hyperparameters for training would do a lot for this project. There are a couple of hyperparameters in particular that I would like to hone in on for future experiments.

In the engineering of text-to-speech systems, the choice of specific audio parameters like num_mels and hop_length is very important in determining synthesized speech’s quality. For instance, hop_length affects temporal resolution within audio processing where shorter hop lengths could allow for the generation of more detailed waveforms required in representing the phonetic subtleties present in natural language. While choosing this parameter one should consider balancing between detail and computational efficiency as well as noise inherent to the audio signal.

Similarly, num_mels, which is the number of Mel spectrogram dimensions to be used, also plays a key role in determining spectral resolution within an audio system. The higher the number of MFCs selected, the higher the level of detail regarding the representation of various spectral characteristics exhibited by different parts of any given sound wave; this is something necessary for achieving clarity and naturalness when it comes to speech synthesis. 80 num_mels seems like a bizarre setting since most other examples of this are set as a power of two. But, after testing the parameters several times, this setting seemed to give the best combo of speed and accuracy.

These two are fundamental components in designing a text-to-speech system but they may need adjusting depending on factors such as languages or desired output qualities. We don't necessarily know what those are when it comes to Esperanto, so it's worth exploring. Such decisions are based on these technical considerations which straddle the complex nature of TTS system design.

In the course of this experiment, I learned the ins and outs of how to train a modern, neural TTS model, as well as how to use the UAZ HPC. This subspecialty of natural language processing really is a blast. It’s always so much fun opening the .wav file for the output and seeing what you get! I definitely plan to delve deeper into this realm of NLP and do more experiments in the future.

All code and data associated with this project can be found at [this](https://github.com/pbarrett520/VITS_esperanto) Github repo!