"""rainday URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from app import urls
from django.contrib.auth import views as a_v
from app.utils import create_model

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts", include("django.contrib.auth.urls")),
    path("auth/pw-change", a_v.PasswordChangeView.as_view(template_name="pw-change.html"), name="pw-change"),
    path("", include(urls)),
]

create_model() 