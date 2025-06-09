from django.urls import path
from .views import Home, Index, Contact, sign_view

urlpatterns = [
    path('', Home.as_view(), name="Home"),
    path('about/', Index.as_view(), name="About"),
    path('contact/', Contact.as_view(), name="Contact"),
    path('sign/', sign_view, name="Sign")
]