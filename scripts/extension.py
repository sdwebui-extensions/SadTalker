import os, sys
from pathlib import Path
import tempfile
import gradio as gr
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call
from modules.shared import opts, OptionInfo
from modules import shared, paths, script_callbacks
import launch
import glob
from huggingface_hub import snapshot_download
from src.utils.init_path import init_path
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
import torch, uuid, shutil
from pydub import AudioSegment
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data

def mp3_to_wav(mp3_filename,wav_filename,frame_rate):
    mp3_file = AudioSegment.from_file(file=mp3_filename)
    mp3_file.set_frame_rate(frame_rate).export(wav_filename,format="wav")



def check_all_files_safetensor(current_dir):
    kv = {
        "SadTalker_V0.0.2_256.safetensors": "sadtalker-256",
        "SadTalker_V0.0.2_512.safetensors": "sadtalker-512",
        "mapping_00109-model.pth.tar" : "mapping-109" ,
        "mapping_00229-model.pth.tar" : "mapping-229" ,
    }

    if not os.path.isdir(current_dir):
        return False
    
    dirs = os.listdir(current_dir)

    for f in dirs:
        if f in kv.keys():
            del kv[f]

    return len(kv.keys()) == 0

def check_all_files(current_dir):
    kv = {
        "auido2exp_00300-model.pth": "audio2exp",
        "auido2pose_00140-model.pth": "audio2pose",
        "epoch_20.pth": "face_recon",
        "facevid2vid_00189-model.pth.tar": "face-render",
        "mapping_00109-model.pth.tar" : "mapping-109" ,
        "mapping_00229-model.pth.tar" : "mapping-229" ,
        "wav2lip.pth": "wav2lip",
        "shape_predictor_68_face_landmarks.dat": "dlib",
    }

    if not os.path.isdir(current_dir):
        return False
    
    dirs = os.listdir(current_dir)

    for f in dirs:
        if f in kv.keys():
            del kv[f]

    return len(kv.keys()) == 0

    

def download_model(local_dir='./checkpoints'):
    REPO_ID = 'vinthony/SadTalker'
    snapshot_download(repo_id=REPO_ID, local_dir=local_dir, local_dir_use_symlinks=False)

def get_source_image(image):   
        return image

def get_img_from_txt2img(x):
    talker_path = Path(shared.cmd_opts.data_dir) / "outputs"
    imgs_from_txt_dir = str(talker_path / "txt2img-images/")
    imgs = glob.glob(imgs_from_txt_dir+'/*/*.png')
    imgs.sort(key=lambda x:os.path.getmtime(os.path.join(imgs_from_txt_dir, x)))
    img_from_txt_path = os.path.join(imgs_from_txt_dir, imgs[-1])
    return img_from_txt_path, img_from_txt_path

def get_img_from_img2img(x):
    talker_path = Path(shared.cmd_opts.data_dir) / "outputs"
    imgs_from_img_dir = str(talker_path / "img2img-images/")
    imgs = glob.glob(imgs_from_img_dir+'/*/*.png')
    imgs.sort(key=lambda x:os.path.getmtime(os.path.join(imgs_from_img_dir, x)))
    img_from_img_path = os.path.join(imgs_from_img_dir, imgs[-1])
    return img_from_img_path, img_from_img_path
 
def get_default_checkpoint_path():
    # check the path of models/checkpoints and extensions/
    checkpoint_path = Path(shared.cmd_opts.data_dir) / "models"/ "SadTalker" 
    extension_checkpoint_path = Path(shared.cmd_opts.data_dir) / "extensions"/ "SadTalker" / "checkpoints"

    if check_all_files_safetensor(checkpoint_path):
        # print('founding sadtalker checkpoint in ' + str(checkpoint_path))
        return checkpoint_path

    if check_all_files_safetensor(extension_checkpoint_path):
        # print('founding sadtalker checkpoint in ' + str(extension_checkpoint_path))
        return extension_checkpoint_path
    
    if check_all_files(checkpoint_path):
        # print('founding sadtalker checkpoint in ' + str(checkpoint_path))
        return checkpoint_path

    if check_all_files(extension_checkpoint_path):
        # print('founding sadtalker checkpoint in ' + str(extension_checkpoint_path))
        return extension_checkpoint_path
    if shared.cmd_opts.just_ui:
        checkpoint_path = Path(os.path.dirname(shared.cmd_opts.data_dir)) / "models"/ "SadTalker" 
        extension_checkpoint_path = Path(os.path.dirname(shared.cmd_opts.data_dir)) / "extensions"/ "SadTalker" / "checkpoints"

        if check_all_files_safetensor(checkpoint_path):
            # print('founding sadtalker checkpoint in ' + str(checkpoint_path))
            return checkpoint_path

        if check_all_files_safetensor(extension_checkpoint_path):
            # print('founding sadtalker checkpoint in ' + str(extension_checkpoint_path))
            return extension_checkpoint_path
        
        if check_all_files(checkpoint_path):
            # print('founding sadtalker checkpoint in ' + str(checkpoint_path))
            return checkpoint_path

        if check_all_files(extension_checkpoint_path):
            # print('founding sadtalker checkpoint in ' + str(extension_checkpoint_path))
            return extension_checkpoint_path

    return None



def install():

    kv = {
        "face_alignment": "face-alignment==1.3.5",
        "imageio": "imageio==2.19.3",
        "imageio_ffmpeg": "imageio-ffmpeg==0.4.7",
        "librosa":"librosa==0.8.0",
        "pydub":"pydub==0.25.1",
        "scipy":"scipy==1.8.1",
        "tqdm": "tqdm",
        "yacs":"yacs==0.1.8",
        "yaml": "pyyaml", 
        "av":"av",
        "gfpgan": "gfpgan",
    }

    # # dlib is not necessary currently
    # if 'darwin' in sys.platform:
    #     kv['dlib'] = "dlib"
    # else:
    #     kv['dlib'] = 'dlib-bin'

    # #### we need to have a newer version of imageio for our method.
    # launch.run_pip("install imageio==2.19.3", "requirements for SadTalker")

    for k,v in kv.items():
        if not launch.is_installed(k):
            print(k, launch.is_installed(k))
            launch.run_pip("install "+ v, "requirements for SadTalker")

    if os.getenv('SADTALKER_CHECKPOINTS'):
        print('load Sadtalker Checkpoints from '+ os.getenv('SADTALKER_CHECKPOINTS'))

    elif get_default_checkpoint_path() is not None:
        os.environ['SADTALKER_CHECKPOINTS'] = str(get_default_checkpoint_path())
    else:

        print(
            """"
            SadTalker will not support download all the files from hugging face, which will take a long time.
             
            please manually set the SADTALKER_CHECKPOINTS in `webui_user.bat`(windows) or `webui_user.sh`(linux)
            """
            )
        
        # python = sys.executable

        # launch.run(f'"{python}" -m pip uninstall -y huggingface_hub', live=True)
        # launch.run(f'"{python}" -m pip install --upgrade git+https://github.com/huggingface/huggingface_hub@main', live=True)
        # ### run the scripts to downlod models to correct localtion.
        # # print('download models for SadTalker')
        # # launch.run("cd " + paths.script_path+"/extensions/SadTalker && bash ./scripts/download_models.sh", live=True)
        # # print('SadTalker is successfully installed!')
        # download_model(paths.script_path+'/extensions/SadTalker/checkpoints')
    
 
def on_ui_tabs():
    install()

    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    sys.path.extend([current_dir]) 
    
    repo_dir = os.path.join(current_dir)

    result_dir = opts.sadtalker_result_dir
    os.makedirs(result_dir, exist_ok=True)

    from app_sadtalker import sadtalker_demo  

    if  os.getenv('SADTALKER_CHECKPOINTS'):
        checkpoint_path = os.getenv('SADTALKER_CHECKPOINTS')
    else:
        checkpoint_path = repo_dir+'checkpoints/'

    audio_to_video = sadtalker_demo(checkpoint_path=checkpoint_path, config_path=repo_dir+'src/config', warpfn = wrap_queued_call)
   
    return [(audio_to_video, "SadTalker", "extension")]

from modules.api.models import *  # noqa:F403
from modules.api import api
from fastapi import FastAPI, Body

def controlnet_api(_: gr.Blocks, app: FastAPI):

    @app.post("/sadtalker/test")
    async def detect(
        source_image: str = Body("", title="source_image path"), 
        driven_audio: str = Body("", title="driven_audio path"), 
        preprocess: str = Body("crop", title="image preprocess"), 
        still_mode: bool=Body(False, title="still_mode"),  
        use_enhancer: bool=Body(False, title="use_enhancer"), 
        batch_size: bool=Body(1, title="batch_size"), 
        size: int=Body(256, title="size"), 
        pose_style: int = Body(0, title="pose_style"), 
        exp_scale: float = Body(1.0, title="exp_scale"), 
        use_ref_video: bool = Body(False, title="use_ref_video"),
        ref_video: str = Body(None, title="ref_video path"),
        ref_info: str = Body(None, title="ref_info"),
        use_idle_mode: bool = Body(False, title="use_idle_mode"),
        length_of_audio: int = Body(0, title="length_of_audio"), 
        use_blink: bool = Body(True, title="use_blink"),
        result_dir: str = Body('./results/', title="result_dir"),
        checkpoint_path: str = Body('./checkpoints', title='checkpoint_path'),
        config_path: str = Body('src/config', title='config_path')
    ):
        sadtalker_paths = init_path(checkpoint_path, config_path, size, False, preprocess)
        print(sadtalker_paths)
        if torch.cuda.is_available() :
            device = "cuda"
        else:
            device = "cpu"
            
        audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
        preprocess_model = CropAndExtract(sadtalker_paths, device)
        animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

        time_tag = str(uuid.uuid4())
        save_dir = os.path.join(result_dir, time_tag)
        os.makedirs(save_dir, exist_ok=True)

        input_dir = os.path.join(save_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)

        print(source_image)
        pic_path = os.path.join(input_dir, os.path.basename(source_image)) 
        shutil.move(source_image, input_dir)

        if driven_audio is not None and os.path.isfile(driven_audio):
            audio_path = os.path.join(input_dir, os.path.basename(driven_audio))  

            #### mp3 to wav
            if '.mp3' in audio_path:
                mp3_to_wav(driven_audio, audio_path.replace('.mp3', '.wav'), 16000)
                audio_path = audio_path.replace('.mp3', '.wav')
            else:
                shutil.move(driven_audio, input_dir)

        elif use_idle_mode:
            audio_path = os.path.join(input_dir, 'idlemode_'+str(length_of_audio)+'.wav') ## generate audio from this new audio_path
            from pydub import AudioSegment
            one_sec_segment = AudioSegment.silent(duration=1000*length_of_audio)  #duration in milliseconds
            one_sec_segment.export(audio_path, format="wav")
        else:
            print(use_ref_video, ref_info)
            assert use_ref_video == True and ref_info == 'all'

        if use_ref_video and ref_info == 'all': # full ref mode
            ref_video_videoname = os.path.basename(ref_video)
            audio_path = os.path.join(save_dir, ref_video_videoname+'.wav')
            print('new audiopath:',audio_path)
            # if ref_video contains audio, set the audio from ref_video.
            cmd = r"ffmpeg -y -hide_banner -loglevel error -i %s %s"%(ref_video, audio_path)
            os.system(cmd)        

        os.makedirs(save_dir, exist_ok=True)
        
        #crop image and extract 3dmm from image
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, first_frame_dir, preprocess, True, size)
        
        if first_coeff_path is None:
            raise AttributeError("No face is detected")

        if use_ref_video:
            print('using ref video for genreation')
            ref_video_videoname = os.path.splitext(os.path.split(ref_video)[-1])[0]
            ref_video_frame_dir = os.path.join(save_dir, ref_video_videoname)
            os.makedirs(ref_video_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_video_coeff_path, _, _ =  preprocess_model.generate(ref_video, ref_video_frame_dir, preprocess, source_image_flag=False)
        else:
            ref_video_coeff_path = None

        if use_ref_video:
            if ref_info == 'pose':
                ref_pose_coeff_path = ref_video_coeff_path
                ref_eyeblink_coeff_path = None
            elif ref_info == 'blink':
                ref_pose_coeff_path = None
                ref_eyeblink_coeff_path = ref_video_coeff_path
            elif ref_info == 'pose+blink':
                ref_pose_coeff_path = ref_video_coeff_path
                ref_eyeblink_coeff_path = ref_video_coeff_path
            elif ref_info == 'all':            
                ref_pose_coeff_path = None
                ref_eyeblink_coeff_path = None
            else:
                raise('error in refinfo')
        else:
            ref_pose_coeff_path = None
            ref_eyeblink_coeff_path = None

        #audio2ceoff
        if use_ref_video and ref_info == 'all':
            coeff_path = ref_video_coeff_path # self.audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
        else:
            batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path=ref_eyeblink_coeff_path, still=still_mode, idlemode=use_idle_mode, length_of_audio=length_of_audio, use_blink=use_blink) # longer audio?
            coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

        #coeff2video
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, batch_size, still_mode=still_mode, preprocess=preprocess, size=size, expression_scale = exp_scale)
        return_path = animate_from_coeff.generate(data, save_dir,  pic_path, crop_info, enhancer='gfpgan' if use_enhancer else None, preprocess=preprocess, img_size=size)
        video_name = data['video_name']
        print(f'The generated video is named {video_name} in {save_dir}')

        del preprocess_model
        del audio_to_coeff
        del animate_from_coeff

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        import gc; gc.collect()
        
        return return_path



def on_ui_settings():
    talker_path = Path(paths.script_path) / "outputs"
    section = ('extension', "SadTalker") 
    opts.add_option("sadtalker_result_dir", OptionInfo(str(talker_path / "SadTalker/"), "Path to save results of sadtalker", section=section)) 

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_app_started(controlnet_api)
