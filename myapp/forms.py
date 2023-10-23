from django import forms
from .models import ImageClass

class ImageForm(forms.ModelForm):
    class Meta:
        model = ImageClass
        fields = '__all__'

