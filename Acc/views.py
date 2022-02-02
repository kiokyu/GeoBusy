import self as self
from django.contrib.auth.models import User
from django.forms import fields, forms, PasswordInput
from django.shortcuts import render
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.views import generic


class SignUpView(generic.CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy('login')
    template_name = 'signup.html'


def ChoiceView(request):
    return render(request, 'choice.html', context={})


def SovetView(request):
    return render(request, 'sovetsky.html', context={})


def PervRView(request):
    return render(request, 'pervorechensky.html', context={})


def PervMView(request):
    return render(request, 'pervomaysky.html', context={})


def FrunView(request):
    return render(request, 'frunzensky.html', context={})


def LenView(request):
    return render(request, 'lenensky.html', context={})