from django.shortcuts import render
from app.utils.predict import predict


def index(request):

    context = {
        "hello": "hello world",
        "form_error": "",
        "initial_data": [],
        "health_result":"",
    }

    if request.method == "POST":
        submit_type = request.GET.get("type")

        if submit_type == "predict":
            body_str = request.body.decode('utf-8')
            data = body_str.split('&')

            dataDict = {} 

            for v in data:
                k, v = v.split('=')
                dataDict[k] = v.strip()

            result = predict(dataDict)
            if result == 0:
               context["health_result"] = "High"
            else:
                context["health_result"] = "Low" 

        return render(request, "index.html", context)

    else:
        return render(request, "index.html", context)