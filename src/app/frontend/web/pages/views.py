from django.shortcuts import render
from django.http import HttpResponse
import requests
import apiConstants as api
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import base64
import cv2
# Create your views here.
# pages/views.py

@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):

        model_request = request.GET.get('model', "all")
        print("model_request", model_request)
        

        image = request.FILES['image'].read()

        # API call
        # TODO: not working the endpoint call 
        data = {
            'file': image
        }
        url= api.PREDICT+model_request
        response = requests.post(url,files=data)
        
        # #convert reponse data into json
        response = response.json()

        return JsonResponse(response)

    return JsonResponse({'status': 'error'})


def home(request):
    return render(request, "pages/home.html", {})
