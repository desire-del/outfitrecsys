from django import forms
from .models import Profile, CURRENT_YEAR
import datetime

class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = [
            'image', 'gender', 'birth_year', 'morphology', 
            'preferred_styles', 'preferred_colors', 'avoid_colors',
            'cultural_constraints', 'size_top', 'size_bottom',
            'allergies', 'style_objectives'
        ]
        
        widgets = {
            'image': forms.ClearableFileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'
            }),
            'gender': forms.Select(attrs={
                'class': 'form-select'
            }),
            'birth_year': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 1900,
                'max': CURRENT_YEAR,
                'placeholder': f'Enter year (1900-{CURRENT_YEAR})'
            }),
            'morphology': forms.Select(attrs={
                'class': 'form-select'
            }),
            'preferred_styles': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Enter styles separated by commas (e.g., casual, elegant, minimalist)'
            }),
            'preferred_colors': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Enter colors separated by commas (e.g., black, beige, navy)'
            }),
            'avoid_colors': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 2,
                'placeholder': 'Enter colors to avoid, separated by commas (e.g., white, beige, yellow)'
            }),
            'cultural_constraints': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 2,
                'placeholder': 'E.g., no short clothing, wearing a veil'
            }),
            'size_top': forms.Select(attrs={
                'class': 'form-select'
            }),
            'size_bottom': forms.Select(attrs={
                'class': 'form-select'
            }),
            'allergies': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 2,
                'placeholder': 'List materials to avoid, separated by commas (e.g., laine, polyester)'
            }),
            'style_objectives': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 2,
                'placeholder': 'List your style objectives, separated by commas (e.g., paraître plus pro, oser la couleur)'
            }),
        }
        
        labels = {
            'image': 'Profile Image',
            'gender': 'Gender',
            'birth_year': 'Birth Year',
            'morphology': 'Body Morphology',
            'preferred_styles': 'Preferred Styles',
            'preferred_colors': 'Preferred Colors',
            'avoid_colors': 'Colors to Avoid',
            'cultural_constraints': 'Cultural Constraints',
            'size_top': 'Top Size',
            'size_bottom': 'Bottom Size',
            'allergies': 'Material Allergies',
            'style_objectives': 'Style Objectives',
        }
        
        help_texts = {
            'image': 'Upload a profile picture (optional)',
            'birth_year': 'Enter your birth year',
            'morphology': 'Select your body type (optional)',
            'preferred_styles': 'Enter styles separated by commas (e.g., casual, elegant, minimalist)',
            'preferred_colors': 'Enter colors separated by commas (e.g., black, beige, navy)',
            'avoid_colors': 'Enter colors to avoid, separated by commas (optional)',
            'cultural_constraints': 'Any cultural or religious clothing preferences (optional)',
            'size_top': 'Select your usual letter size for tops',
            'size_bottom': 'Select your usual letter size for bottoms',
            'allergies': 'List any materials to avoid, separated by commas (optional)',
            'style_objectives': 'List your style goals, separated by commas (optional)',
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Make morphology field optional in the form
        self.fields['morphology'].required = False
        self.fields['morphology'].empty_label = "Select morphology (optional)"
        
        # Set required fields
        required_fields = ['gender', 'birth_year', 'preferred_styles', 'preferred_colors', 'size_top', 'size_bottom']
        for field_name in required_fields:
            self.fields[field_name].required = True
            
        # Add asterisk to required field labels
        for field_name, field in self.fields.items():
            if field.required:
                field.label = f"{field.label} *"

    def clean_birth_year(self):
        birth_year = self.cleaned_data.get('birth_year')
        if birth_year:
            current_year = datetime.datetime.now().year
            if birth_year < 1900 or birth_year > current_year:
                raise forms.ValidationError(f'Birth year must be between 1900 and {current_year}')
        return birth_year

    def clean_preferred_styles(self):
        styles = self.cleaned_data.get('preferred_styles', '')
        if styles:
            # Clean up the styles - remove extra spaces and empty items
            styles_list = [style.strip() for style in styles.split(',') if style.strip()]
            if not styles_list:
                raise forms.ValidationError('Please enter at least one preferred style.')
            return ', '.join(styles_list)
        raise forms.ValidationError('Preferred styles are required.')

    def clean_preferred_colors(self):
        colors = self.cleaned_data.get('preferred_colors', '')
        if colors:
            # Clean up the colors - remove extra spaces and empty items
            colors_list = [color.strip() for color in colors.split(',') if color.strip()]
            if not colors_list:
                raise forms.ValidationError('Please enter at least one preferred color.')
            return ', '.join(colors_list)
        raise forms.ValidationError('Preferred colors are required.')

    def clean_avoid_colors(self):
        colors = self.cleaned_data.get('avoid_colors', '')
        if colors:
            # Clean up the colors - remove extra spaces and empty items
            colors_list = [color.strip() for color in colors.split(',') if color.strip()]
            return ', '.join(colors_list)
        return colors

    def clean_allergies(self):
        allergies = self.cleaned_data.get('allergies', '')
        if allergies:
            # Clean up the allergies - remove extra spaces and empty items
            allergies_list = [allergy.strip() for allergy in allergies.split(',') if allergy.strip()]
            return ', '.join(allergies_list)
        return allergies

    def clean_style_objectives(self):
        objectives = self.cleaned_data.get('style_objectives', '')
        if objectives:
            # Clean up the objectives - remove extra spaces and empty items
            objectives_list = [obj.strip() for obj in objectives.split(',') if obj.strip()]
            return ', '.join(objectives_list)
        return objectives