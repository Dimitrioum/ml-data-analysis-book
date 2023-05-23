from django import forms

class DocumentForm(forms.Form):
    docfile = forms.FileField(label='Выберите файлы', widget=forms.ClearableFileInput(attrs={'multiple': False}))

class DocumentFormCarPlates(forms.Form):
    car_plates_file_input = forms.FileField(label='Выберите файлы', widget=forms.ClearableFileInput(attrs={'multiple': False}))

class LoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)
