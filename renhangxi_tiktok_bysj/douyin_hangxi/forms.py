from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import PasswordChangeForm


class UserProfileForm(forms.ModelForm):
    """
    用户基础信息修改表单
    用于修改用户名、邮箱等非敏感信息
    """

    class Meta:
        model = User
        fields = ['username', 'email']
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '请输入新的用户名'
            }),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'placeholder': '请输入邮箱地址'
            }),
        }

    def clean_email(self):
        """
        自定义邮箱验证：确保邮箱格式正确且未被其他用户占用（可选）
        """
        email = self.cleaned_data.get('email')
        if email and User.objects.filter(email=email).exclude(username=self.instance.username).exists():
            raise forms.ValidationError("该邮箱已被其他账号使用。")
        return email


class CustomPasswordChangeForm(PasswordChangeForm):
    """
    自定义密码修改表单
    继承自 Django 原生 PasswordChangeForm，自动包含旧密码验证和新密码强度校验
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 为所有字段添加 Bootstrap 样式类
        for field in self.fields.values():
            field.widget.attrs.update({'class': 'form-control'})