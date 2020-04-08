from django.shortcuts import render
from django.contrib.auth import authenticate, login, logout


# Create your views here.
def login_view(request):
    if not request.POST:
        return render(request, 'account/login.html')
    if request.POST:
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if not user:
            context = {
                'error': "Error login or password"
            }
            #logger.error("User :" + username + " try to login")
            return render(request, 'account/login.html', context)
        else:
            if user is not None:
                login(request, user)
                #logger.info("User :" + username + " login")
                return render(request, "data_set/model_training.html")
            else:
                #logger.error("User :" + username + " try to login")
                return render(request, "account/login.html")


def logout_view(request):
    logout(request)
    return render(request, "data_set/model_training.html")


def register(request):
    return render(request, "account/register.html")


def cabinet(request):
    return render(request, "account/cabinet.html")


def change_password(request):
    return render(request, "data_set/model_training.html")


def change_info(request):
    return render(request, "data_set/model_training.html")