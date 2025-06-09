from django.contrib import admin
from .models import Profile, WardrobeItem, Outfit, Feedback

admin.site.register((Profile, WardrobeItem, Outfit, Feedback))