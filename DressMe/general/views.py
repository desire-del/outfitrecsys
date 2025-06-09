from django.shortcuts import render, redirect
from django.views.generic import TemplateView
from django.contrib.auth import login, authenticate
from django.contrib.auth.models import User
from .forms import LoginForm, RegisterForm

# Create your views here.

class Home (TemplateView) : 
    template_name = "general/index.html"

class Index (TemplateView) : 
    template_name = "general/about.html"

class Contact (TemplateView) : 
    template_name = "general/contact.html"

def sign_view (request) : 
    error = None
    login_form = LoginForm()
    register_form = RegisterForm()
    if request.method == "POST" : 
        form_type = request.POST.get("form_type")
        if form_type == "sign_in": 
            login_data = {
                key: value for key, value in request.POST.items()
                if key in ['username', 'password']
            }
            login_form = LoginForm(login_data)
            if login_form.is_valid():
                username = login_form.cleaned_data['username']
                password = login_form.cleaned_data['password']

                user = authenticate(request, username=username, password=password)
                if user is not None:
                    login(request, user)
                    return redirect('Home')
                else:
                    error = "Invalid username or password"
        elif form_type == "sign_up" :
            register_data = {
                key: value for key, value in request.POST.items()
                if key in ['username', 'email', 'password1', 'password2']
            }
            register_form = RegisterForm(register_data)
            if register_form.is_valid():
                username = register_form.cleaned_data['username']
                request.session['temp_registration_data'] = {
                    'username': username,
                    'email': register_form.cleaned_data['email'],
                    'password': register_form.cleaned_data['password1']
                }
                # username = request.POST.get("username")
                # email = request.POST.get("email")
                # password1 = request.POST.get("password1")
                if User.objects.filter(username=username).exists():
                    error = "Username already exists"
                else:
                    return redirect("Complete_profile")
                    # user = User.objects.create_user(username=username, email=email, password=password1)
                    # login(request, user)
                    # return redirect('Home')
            else : 
                error = "Invalid registration data"
    return render(request, 'general/sign.html', {'login_form': login_form, 'register_form': register_form, 'error' : error })