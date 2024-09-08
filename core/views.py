# views.py

from django.shortcuts import render

def index(request):
    return render(request, 'index.html')

def another_page(request):
    return render(request, 'another_page.html')
def home(request):
    return render(request, 'home.html')
