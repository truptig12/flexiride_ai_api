from django.contrib import admin
from django.urls import path
from django.urls import re_path

from django.conf.urls import include
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    re_path(r'^image/', include('flexirideapi.urls')),
	]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
