from django.urls import path
from pages import views

urlpatterns = [
    path("", views.home, name='home'),
    path("upload_image", views.upload_image, name='upload_image'),
]

