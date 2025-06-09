from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from .choices import Gender, Morphology, Size, Season, Occasion, Cuts
import datetime

CURRENT_YEAR = datetime.date.today().year


class Profile(models.Model) : 
    user = models.OneToOneField(to=User, on_delete=models.CASCADE)
    image = models.ImageField(
        upload_to="profile_pics/", 
        blank=True, 
        null=False, 
        default='profile_pics/default.png'
    )
    gender = models.CharField(
        max_length=1,
        choices=Gender.choices,
        default=Gender.UNSPECIFIED,
        blank=False, 
        null=False
    )
    birth_year = models.IntegerField(
        validators=[
            MinValueValidator(1900),
            MaxValueValidator(CURRENT_YEAR)
        ],
        blank=False,
        null=False
    )
    morphology = models.CharField (
        max_length=20,
        choices= Morphology.choices,
        blank=True,
        null=True,
    )
    preferred_styles = models.TextField(
        help_text='Enter styles separated by commas (e.g., casual, elegant, minimalist)',
        blank=False, 
        null=False
    )
    preferred_colors = models.TextField(
        help_text='Enter colors separated by commas (e.g., black, beige, navy)',
        blank=False, 
        null=False
    )
    avoid_colors = models.TextField(
        blank=True,
        null=False,
        help_text="Enter colors to avoid, separated by commas (e.g., White, beige, yellow)"
    )
    cultural_constraints = models.TextField(
        blank=True,
        null=False,
        help_text="E.g., no short clothing, wearing a veil"
    )
    size_top = models.CharField(
        max_length=3,
        choices=Size.choices,
        blank=False,
        null=False,
        help_text="Select your usual letter size for the top of your body"
    )
    size_bottom = models.CharField(
        max_length=3,
        choices=Size.choices,
        blank=False,
        null=False,
        help_text="Select your usual letter size for the bottom of your body"
    )
    allergies = models.TextField(
        blank=True,
        null=False,
        help_text='List any materials to avoid, separated by commas (e.g., "laine, polyester")'
    )
    style_objectives = models.TextField(
        blank=True,
        null=False,
        help_text='List your style objectives, separated by commas (e.g., "paraître plus pro, oser la couleur")'
    )
    def __str__(self : "Profile"):
        return f"{self.user.username}'s Profile"
    
    @staticmethod 
    def _split_csv(text: str) : 
        if not text : 
            return []
        return [item.strip() for item in text.split(",") if item.strip()]
    
    @staticmethod
    def _join_csv(lst : list) : 
        if isinstance(lst, list) : 
            return ', '.join(lst)
        return lst
    
    @property 
    def preferred_styles_list(self : "Profile") : 
        return self._split_csv(self.preferred_styles)
    
    @preferred_styles_list.setter
    def preferred_styles_list(self : "Profile", value : list) : 
        self.preferred_styles = self._join_csv(value)

    @property
    def preferred_colors_list(self):
        return self._split_csv(self.preferred_colors)

    @preferred_colors_list.setter
    def preferred_colors_list(self, value):
        self.preferred_colors = self._join_csv(value)

    @property
    def avoid_colors_list(self):
        return self._split_csv(self.avoid_colors)

    @avoid_colors_list.setter
    def avoid_colors_list(self, value):
        self.avoid_colors = self._join_csv(value)

    @property
    def allergies_list(self):
        return self._split_csv(self.allergies)

    @allergies_list.setter
    def allergies_list(self, value):
        self.allergies = self._join_csv(value)

    @property
    def style_objectives_list(self):
        return self._split_csv(self.style_objectives)

    @style_objectives_list.setter
    def style_objectives_list(self, value):
        self.style_objectives = self._join_csv(value)

class WardrobeItem(models.Model):
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE, related_name="wardrobe_items")
    name = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text="Give a name or short description for this clothing item (e.g., Blue Denim Jacket)."
    )
    image = models.ImageField(upload_to="wordrobe_pics/", blank=False, null=False)
    type = models.CharField(
        max_length=50,
        blank=False, 
        null=False,
        help_text="Enter the type of clothing shown in the picture you are providing (e.g., pants, shirt, dress, etc.)."
    )
    color = models.CharField(
        max_length=50,
        blank=False, 
        null=False,
        help_text="Enter the color of clothing shown in the picture you are providing (e.g., black, red, etc.)"
    )
    material = models.CharField(
        max_length=100,
        blank=False, 
        null=False,
        help_text="Enter the material of the clothing item (e.g., cotton, leather, wool)"
    )
    season = models.CharField (
        max_length=10,
        choices=Season.choices,
        blank=True,
        null=True,
        help_text="Select the season this clothing is suitable for (e.g., Summer, Winter)."
    )
    occasion = models.CharField (
        max_length=10,
        choices=Occasion.choices,
        blank=True,
        null=True,
        help_text="Select the occasion this clothing is typically used for (e.g., casual, formal, sport)."
    )
    cut = models.CharField(
        max_length=10,
        choices=Cuts.choices,
        blank=False,
        null=False,
        help_text="Select the cut (fit) of the clothing item, such as slim, regular, or oversize."
    )
    size = models.CharField(
        max_length=3,
        choices=Size.choices,
        blank=False,
        null=False,
        help_text="Select the size of the clothing item (e.g., S, M, L, 38, 40)."
    )
    brand = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text="Specify the brand of the clothing item, if known."
    )
    style_tags = models.TextField(
        blank=False,
        null=False,
        help_text="Enter style tags separated by commas (e.g., minimalist, streetwear, vintage)."
    )
    date_added = models.DateTimeField(auto_now_add=True)
    favorite = models.BooleanField(
        default=False,
        help_text="Mark this item as a favorite."
    )

    def __str__(self):
        return self.name or f"Item #{self.pk} ({self.type})"

    @property
    def style_tags_list(self):
        if not self.style_tags:
            return []
        return [tag.strip() for tag in self.style_tags.split(",") if tag.strip()]

    @style_tags_list.setter
    def style_tags_list(self, value):
        if isinstance(value, list):
            self.style_tags = ", ".join(value)
        else:
            self.style_tags = value

class Outfit(models.Model):
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE, related_name="outfits")
    name = models.CharField(max_length=100, blank=True, null=True)
    items = models.ManyToManyField(WardrobeItem, related_name="outfits")
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name or f"Outfit #{self.pk}"

class Feedback(models.Model):
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE, related_name="feedbacks")
    outfit = models.ForeignKey(Outfit, on_delete=models.CASCADE, related_name="feedbacks")
    rating = models.PositiveSmallIntegerField()
    comment = models.TextField(blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Feedback by {self.profile.user.username} on Outfit #{self.outfit.pk}"