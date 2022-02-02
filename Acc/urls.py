from django.urls import path
from .views import *

urlpatterns = [
    path('signup/', SignUpView.as_view(), name='signup'),
    path('choice/', ChoiceView, name='choice'),
    path('sovetsky/', SovetView, name='sovetsky'),
    path('pervomaysky/', PervMView, name='pervomaysky'),
    path('pervorechensky/', LenView, name='pervorechensky'),
    path('leninsky/', PervRView, name='leninsky'),
    path('frunzensky/', FrunView, name='frunzensky'),
]
