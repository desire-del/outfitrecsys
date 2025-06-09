from django.urls import path
from .views import wardrobe_view

urlpatterns = [
    path("wardrobe", wardrobe_view, name="wardrobe")
]