from django.conf import settings
from django.conf.urls.static import static

from django.contrib import admin
from django.urls import path
from myapp.views import home, Vgg, Cnn

urlpatterns = [
    path("", home, name="home"),
    path("vgg/", Vgg, name="vgg"),
    path("inception/", Cnn, name="Cnn"),
    path('admin/', admin.site.urls),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
