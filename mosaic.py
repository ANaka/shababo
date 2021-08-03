import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from itertools import product

@dataclass
class MosaicBlock:
    ix_start: int
    ix_end: int
    iy_start: int
    iy_end: int
    filled: bool = False

    @classmethod
    def from_pixels_per_block(
        cls,
        row,
        col,
        pixels_per_block_x=128,
        pixels_per_block_y=128,
    ):
        """
        Create a mosaic from pixels per block.
        """
        
        ix_start = col * pixels_per_block_x
        ix_end = (col + 1 ) * pixels_per_block_x
        iy_start = row * pixels_per_block_y
        iy_end = (row + 1 ) * pixels_per_block_y
        return cls(
            ix_start=ix_start,
            ix_end=ix_end,
            iy_start=iy_start,
            iy_end=iy_end,
        )

    


class MosaicGrid(object):
    """
    Class for creating a mosaic grid from a set of images. Images can be set to 3 different sizes.
    """
    def __init__(
        self,
        image_dir,
        n_rows=11,
        n_cols=7,
        pixels_per_block_x=128,
        pixels_per_block_y=128,
        glob_str='*.png'
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.pixels_per_block_x = pixels_per_block_x
        self.pixels_per_block_y = pixels_per_block_y
        self.image_dir = Path(image_dir)

        self.n_pixels_x = self.n_cols * self.pixels_per_block_x
        self.n_pixels_y = self.n_rows * self.pixels_per_block_y
        self.mosaic = np.zeros((self.n_pixels_y, self.n_pixels_x))
        self.blocks = {iy:{ix:{} for ix in range(self.n_cols)} for iy in range(self.n_rows)}
        for irow in range(self.n_rows):
            for icol in range(self.n_cols):
                self.blocks[irow][icol] = MosaicBlock.from_pixels_per_block(
                    row=irow,
                    col=icol,
                    pixels_per_block_x=self.pixels_per_block_x,
                    pixels_per_block_y=self.pixels_per_block_y,
                )

        self.files = sorted(list(self.image_dir.glob(glob_str)))

    @property
    def unfilled_block_coords(self):
        return [(row,col) for row, col in product(range(self.n_rows), range(self.n_cols)) if not self.blocks[row][col].filled]

    def get_pixel_indices_from_blocks(self, start_row, start_col, size=1):
        end_row = start_row + size
        end_col = start_col + size

        current_blocks = []
        for irow, icol in product(range(start_row, end_row), range(start_col, end_col)):
            current_blocks.append(self.blocks[irow][icol])

        ix_start = np.min([block.ix_start for block in current_blocks])
        ix_end = np.max([block.ix_end for block in current_blocks])
        iy_start = np.min([block.iy_start for block in current_blocks])
        iy_end = np.max([block.iy_end for block in current_blocks])
        return ix_start, ix_end, iy_start, iy_end

    def add_image(self, image_name, start_row, start_col, size=1, invert=False):
        """
        Add an image to the mosaic.
        """
        ix_start, ix_end, iy_start, iy_end = self.get_pixel_indices_from_blocks(start_row=start_row, start_col=start_col, size=size)
        im = Image.open(self.image_dir.joinpath(image_name))
        im = im.resize((self.pixels_per_block_x*size, self.pixels_per_block_y*size)).convert('L')
        self.mosaic[iy_start:iy_end, ix_start:ix_end] = np.array(im)

        end_row = start_row + size
        end_col = start_col + size
        for irow, icol in product(range(start_row, end_row), range(start_col, end_col)):
            self.blocks[irow][icol].filled = True

        if invert:
            self.invert_block(start_row=start_row, start_col=start_col, size=size)

    def invert_block(self, start_row, start_col, size=1):
        """
        Invert a block.
        """
        ix_start, ix_end, iy_start, iy_end = self.get_pixel_indices_from_blocks(start_row=start_row, start_col=start_col, size=size)
        self.mosaic[iy_start:iy_end, ix_start:ix_end] = 255 - self.mosaic[iy_start:iy_end, ix_start:ix_end]
        

    def show_mosaic(self):
        return Image.fromarray(self.mosaic).convert('L')
