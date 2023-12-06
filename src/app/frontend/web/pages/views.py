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
        try:
            model_request = request.GET.get('model', "all")
            image = request.FILES['image'].read()
            data = {
                'file': image
            }
            url= api.PREDICT+model_request
            response = requests.post(url,files=data)

            if response.status_code != 200:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Error from server: {response.status_code}',
                    'detail': response.json()
                }, status=response.status_code, )

            response_data = response.json()
            return JsonResponse(response_data)

        
        except requests.exceptions.RequestException as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

    return JsonResponse({'status': 'error'})


def home(request):
    return render(request, "pages/home.html", {})
