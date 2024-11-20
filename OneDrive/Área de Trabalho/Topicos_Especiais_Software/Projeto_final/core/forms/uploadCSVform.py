from django import forms

class UploadCSVForm(forms.Form):
    file = forms.FileField(label="Selecione um arquivo CSV")
