'''
Created on 08-Nov-2018

@author: anu
'''
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import permission_classes
from rest_framework import permissions
from .SpeechEmotion import Emotion
from .amplitude import match_target_amplitude
from pydub import AudioSegment
import os
import uuid
import io


path = os.getcwd() + '/SpeechEmotion/SpeechEmotion'

@permission_classes((permissions.AllowAny,))
class EmotionPredict(viewsets.ViewSet):
    def create(self, request):
        files = request.FILES.getlist('file')
        if len(files) > 0:
            file = files[0]
            filename = file.name
            format = filename.split('.')[-1:][0]
            try:
                current_path = path + '/temp'
                for chunk in file.chunks():
                    Audio = AudioSegment.from_file(io.BytesIO(chunk), format=format)
                    normalized_audio = match_target_amplitude(Audio, -20.0)
                    file_path = os.path.join(current_path, 'emotion' + uuid.uuid4().hex) 
                    normalized_audio.export(file_path+'.wav', format="wav")
                file_name = file_path + '.wav'
                emotionprediction = Emotion(file_name)
                if os.path.exists(file_name):
                    os.remove(file_name)
                return Response(emotionprediction)
            except:
                result = {}
                result['Message'] = 'Please enter voice note'
                if os.path.exists(file_name):
                    os.remove(file_name)
                return Response(result)
        else:
            result = {}
            result['Message'] = 'Please enter your voice note' 
            return Response(result)
