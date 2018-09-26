import cairo
import gi
gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Pango, PangoCairo
from PIL import Image
import numpy as np


class FontImageRender:
    def __init__(self, font, width, height):
        self.font = font
        self.width, self.height = width, height
    
    def create_params(self):
        self.surface = cairo.ImageSurface(cairo.FORMAT_A8, self.width, self.height)
        self.context = cairo.Context(self.surface)
        self.pc = PangoCairo.create_context(self.context)
        self.layout = PangoCairo.create_layout(self.context)
        
        self.layout.set_font_description(Pango.FontDescription(self.font))
        self.font_desc = Pango.font_description_from_string(self.font)
    
    def __call__(self, text):
        self.create_params()
        self.layout.set_text(text, -1)
        stroke_rect, _ = self.layout.get_pixel_extents()
        
        # Applies a fit-height variation.
        font_desc = self.layout.get_font_description()
        font_desc.set_size(int(font_desc.get_size() * 0.8 * self.height/stroke_rect.height ) )
        self.layout.set_font_description(font_desc)
        stroke_rect, _ = self.layout.get_pixel_extents()

        PangoCairo.show_layout(self.context, self.layout)
        data = self.surface.get_data()
        #data = np.frombuffer(data, dtype=np.uint8).reshape(( targetH, self.width ))[:, :stroke_rect.width]
        data = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width))[:, :stroke_rect.width]
        data = np.invert(data)
        return data
