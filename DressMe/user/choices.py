from django.db import models

class Gender (models.TextChoices) :
    Male = 'M', "Male"
    Female = 'F', "Female"
    Non_Binary = 'N', "Non-binary"
    Other = 'O', "Other"
    UNSPECIFIED = 'U', "Prefer not to say"

class Morphology (models.TextChoices) : 
    RECTANGLE = 'rectangle', 'Rectangle'
    PYRAMID = 'pyramid', 'Pyramid'
    OVAL = 'oval', 'Oval'
    HOURGLASS = 'hourglass', 'Hourglass'
    ROUND = 'round', 'Round'

class Size(models.TextChoices):
    SIZE_34 = "34", "34 (XS)"
    SIZE_36 = "36", "36 (S)"
    SIZE_38 = "38", "38 (M)"
    SIZE_40 = "40", "40 (L)"
    SIZE_42 = "42", "42 (XL)"
    SIZE_44 = "44", "44 (XXL)"
    SIZE_46 = "46", "46 (XXL)"

class Season (models.TextChoices) : 
    Spring = 'spring', 'Spring'
    Summer = 'summer', 'Summer'
    Autumn = 'autumn', 'Autumn'
    Winter = 'winter', 'Winter'

class Occasion (models.TextChoices) : 
    Casual = 'casual', 'Casual'
    Formal = 'formal', 'Formal'
    Sport = 'sport', 'Sport'
    Party = 'party', 'Party'
    Work = 'work', 'Work'
    Other = 'other', 'Other'

class Cuts (models.TextChoices) : 
    Slim = 'slim', 'Slim'
    Regular = 'regular', 'Regular'
    Oversize = 'oversize', 'Oversize'
    Relaxed = 'relaxed', 'Relaxed'
    Fitted = 'fitted', 'Fitted'