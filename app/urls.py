from django.urls import path

from app import views as v

urlpatterns = [
    path("", v.index, name="home"),
]
