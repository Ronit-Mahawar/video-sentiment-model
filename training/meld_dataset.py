from torch.utils.data import Dataset,DataLoader
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch
import subprocess
import torchaudio 
import soundfile as sf
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#test
class MELDDataset(Dataset):
    def __init__(self,csv_path,video_dir):
        self.data=pd.read_csv(csv_path)
        
        self.video_dir=video_dir
        self.tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')
       #Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear
        self.emotion_map={
            'anger': 0,'disgust': 1,'sadness': 5,'joy': 3,'neutral': 4,'surprise': 6,'fear': 2
        }
        self.sentiment_map={
            'negative': 0,'neutral': 1,'positive': 2
        }
    
    def _load_video_frames(self,video_path):
        cap=cv2.VideoCapture(video_path)
        frames=[]

        try:
            if not cap.isOpened():

                raise ValueError(f"Video not found:{video_path}")

            #try and read first frame to validate video
            ret,frame=cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found:{video_path}")
            
            #reset index to not skip first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            while len(frames) < 30 and cap.isOpened():
                ret,frame=cap.read()
                if not ret:
                    break
                frame=cv2.resize(frame,(224,224))
                frame=frame/255.0 
                frames.append(frame)
            
            #padding truncate
            if len(frames)<30:
                frames+=[np.zeros_like(frames[0])] * (30 - len(frames))
            else:
                frames=frames[:30 ]
            
            #before permute [frames,height,width,channles]
            #after permute [frames,channel,height,width]

            return torch.FloatTensor(np.array(frames)).permute(0,3,1,2)
        
        except Exception as e:
            raise ValueError(f"Video error:{str(e)}")    
        finally:
            cap.release()

        if len(frames)==0:
            raise ValueError("No frames could be extracted")



    def _extract_audio_features(self,video_path):
        audio_path=video_path.replace('.mp4','.wav')

        try:
            subprocess.run([
                'ffmpeg',
                '-i',video_path,
                '-vn',
                '-acodec','pcm_s16le',
                '-ar','16000',
                '-ac','1',
                audio_path
            ],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

            waveform, sample_rate=sf.read(audio_path)

            # if sample_rate !=16000:
            #     resampler=torchaudio.transforms.Resample(sample_rate,16000)
            #     waveform=resampler(waveform)
              # Convert to tensor
            waveform = torch.tensor(waveform, dtype=torch.float32)

            if waveform.ndim == 2:  # stereo → mono
                waveform = waveform.mean(dim=1)

            waveform = waveform.unsqueeze(0)  # [1, samples]
            
            mel_spectogram=torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )
            mel_spec=mel_spectogram(waveform)


            #Normalize
            mel_spec=(mel_spec -mel_spec.mean()) / mel_spec.std()

            if mel_spec.size(2) < 300:
                padding=300 - mel_spec.size(2)
                mel_spec=torch.nn.functional.pad(mel_spec,(0, padding))
            else:
                mel_spec=mel_spec[:, :, :300]
            
            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio errro:{str(e)}")
 
        except Exception as e:
            raise ValueError(f"Audio error {str(e)}")
        finally:
             if os.path.exists(audio_path):
                 os.remove(audio_path)
    
        


    
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx=idx.item()
        row=self.data.iloc[idx]

        try:
            video_filename=f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""
            path=os.path.join(self.video_dir,video_filename)
            video_path_exists=os.path.exists(path)
            
            if video_path_exists==False:
                raise FileNotFoundError(f"No video found for filename: {path}")
            
            text_inputs=self.tokenizer(row['Utterance'],
                                    padding='max_length',
                                    truncation=True,
                                    max_length=128,
                                    return_tensors='pt')
            
            #emmotion and sentiment map
            emotion_label = self.emotion_map[row['Emotion'].lower()]
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]

            
            video_frames=self.load_video_frames(path)
            audio_features= self._extract_audio_features(path)
            
            return {
                'text_inputs':{
                    'input_ids': text_inputs['input_ids'].squeeze(),
                    'attention_mask':text_inputs['attention_mask'].squeeze()
                },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion_label': torch.tensor(emotion_label),
                'sentiment_label':torch.tensor(sentiment_label)
            }        
        except Exception as e:
            raise ValueError(f"Error Processiont {path}: {str(e)}")
            return None

def collate_fn(batch):
    #filter out  None samples
    batch = list(filter(None,batch))
    return torch.utils.data.dataloader.default_collate(batch)
    

def prepare_dataloader(train_csv,train_video_dir,
                        dev_csv,dev_video_dir,
                        test_csv,test_video_dir,batch_size=32):
    train_dataset = MELDDataset(train_csv,train_video_dir)  
    dev_dataset = MELDDataset(dev_csv,dev_video_dir)
    test_dataset = MELDDataset(test_cst,test_video_dir)

    train_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=collate_fn)
    
    dev_loader = DataLoader(dev_dataset,
                                batch_size=batch_size,
                                collate_fn=collate_fn)
    
    test_loader = DataLoader(test_dataset,
                                batch_size=batch_size,
                                collate_fn=collate_fn)
    
    return train_loader,dev_loader,test_loader
    


if __name__== "__main__":
    train_loader,dev_loader,test_loader= prepare_dataloader(
        '../dataset/train/train_sent_emo.csv','../dataset/train/train_splits',
        '../dataset/dev/dev_sent_emo.csv','../dataset/dev/dev_splits_complete',
        '../dataset/test/test_sent_emo.csv','../dataset/test/output_repeated_splits_test',

    )

    for batch in train_loader:
        print(batch['text_inputs'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        break

     
        


 

