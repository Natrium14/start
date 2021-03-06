from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout


# Create your views here.
from account.forms import RegisterForm


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
                return redirect("/data_set/index")
            else:
                #logger.error("User :" + username + " try to login")
                return render(request, "account/login.html")


def logout_view(request):
    logout(request)
    return redirect("/")


def register(response):
    try:
        if response.method == "POST":
            form = RegisterForm(response.POST)
            if form.is_valid():
                form.save()

            return redirect("/data_set/index")
        else:
            form = RegisterForm()

        return render(response, "account/register.html", {"form":form})
    except:
        return render("error/error404.html")


def cabinet(request):
    return render(request, "account/cabinet.html")


def change_password(request):
    if request.POST:
        current_user = request.user
        print(request.user)
        current_user.set_password(request.POST['password'])
        current_user.save()
        return render(request, "account/cabinet.html")
    return render(request, "account/cabinet.html")


def change_info(request):
    return render(request, "data_set/index.html")