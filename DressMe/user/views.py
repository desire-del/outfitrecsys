from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import login
from .forms import ProfileForm

def profile_view (request) : 
    if 'temp_registration_data' not in request.session:
        return redirect('Sign')
    
    if request.method == "POST" : 
        profile_form = ProfileForm(request.POST, request.FILES)
        if profile_form.is_valid() : 
            data = request.session.pop('temp_registration_data')
            user = User.objects.create_user(
                username=data['username'],
                email=data['email'],
                password=data['password']
            )
            profile = profile_form.save(commit=False)
            profile.user = user
            profile.save()

            login(request=request, user=user)
            return redirect("Home")
    else : 
        profile_form = ProfileForm()
    return render(request=request, template_name="user/profile.html", context={'profile_form': profile_form})