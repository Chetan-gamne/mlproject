from django.shortcuts import render
from pickle import load
from django.views.decorators.csrf import csrf_exempt

class Predict_model:
    def __init__(self,age,cgp,chol,sBP,dBP,BMI,hrtRate,glucose):
        self.age = age
        self.cgp = cgp
        self.chol = chol
        self.sBP = sBP
        self.dBP = dBP
        self.BMI = BMI
        self.hrtRate = hrtRate
        self.glucose = glucose
    
    def solve(self):
        x_test = [[self.age,self.cgp,self.chol,self.sBP,self.dBP,self.BMI,self.hrtRate,self.glucose]]
        model = load(open('svm_model.pkl','rb'))
        scaler = load(open('scaler.pkl','rb'))
        X_test_scaled = scaler.transform(x_test)
        prediction = model.predict(X_test_scaled)
        if prediction[0]==0:
            return "Your Safe"
        else:
            return "Your not in Safe Zone Take Proper Precautions"
        
def home(request):
    return render(request,'home.html')

@csrf_exempt
def add(request):
    age = float(request.POST['Age'])
    cgp = float(request.POST['cgp'])
    chol = float(request.POST['chol'])
    sBP = float(request.POST['sBP'])
    dBP = float(request.POST['dBP'])
    BMI = float(request.POST['BMI'])
    hrtRate = float(request.POST['hrtRate'])
    glucose = float(request.POST['glucose'])
    obj1 = Predict_model(age,cgp,chol,sBP,dBP,BMI,hrtRate,glucose)
    # obj1 = Solve(num1,num2)
    return render(request,'result.html',{'total':obj1.solve()})