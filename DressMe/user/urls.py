from django.urls import path
from .views import profile_view

urlpatterns = [
    path("complete_registration", profile_view, name="Complete_profile")
]