from pydantic import BaseModel


class WardrobeItem(BaseModel):
    """
    A class representing a wardrobe item.

    Attributes:
        id (int): The unique identifier for the wardrobe item.
        name (str): The name of the wardrobe item.
        category (str): The category of the wardrobe item (e.g., shirt, pants, etc.).
        color (str): The color of the wardrobe item.
        size (str): The size of the wardrobe item.
        image_url (str): The URL of the image representing the wardrobe item.
    """

    id: str
    image_url: str
    category: str
    color: str
    material: str
    