from typing import List
from shapely.geometry import MultiPolygon, Polygon
from pystac_client import Client
import numpy as np
#change this
from scripts.forrest_config import Config

catalog_element84 = Client.open(Config.ELEMENT84_STAC_URL)

def compute_grids(polygon: Polygon, grid_size: float) -> List[Polygon]:
    """Divide the polygon into grid based polygons

    Args:
        polygon (Polygon): Polygon to divide
        grid_size (float): The granularity of the division. Higher the number, bigger the polygons.

    Returns:
        List[Polygon]: Divided polygons
    """
    polygons = []
    min_x, min_y, max_x, max_y = polygon.bounds
    for x in np.arange(min_x, max_x, grid_size):
        for y in np.arange(min_y, max_y, grid_size):
            rectangle = (
                Polygon(
                    [
                        (x, y),
                        (x + grid_size, y),
                        (x + grid_size, y + grid_size),
                        (x, y + grid_size),
                    ]
                )
                .intersection(polygon)
                .buffer(1e-4)
            )
            if rectangle.area == 0:
                continue
            polygons.append(rectangle)

    return polygons
