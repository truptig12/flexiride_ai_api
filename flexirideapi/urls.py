from django.contrib import admin
from django.urls import path
from flexirideapi import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('image/', views.ImageAPIView.as_view(),name="Image Upload API"),
]