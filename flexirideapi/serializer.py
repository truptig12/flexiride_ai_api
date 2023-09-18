from rest_framework import serializers


class ImgSerializer(serializers.ModelSerializer):
    image = serializers.ImageField(max_length=None, use_url=True)
    # permission_classes = (permissions.IsAuthenticatedOrReadOnly,)
    class Meta:
        model = Img
        fields = ( 'image')