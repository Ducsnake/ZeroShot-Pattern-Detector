from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


def iter_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[Tile]:
    if tile_size <= 0 or width <= tile_size and height <= tile_size:
        return [Tile(0, 0, width, height)]

    overlap = min(max(0, int(overlap)), max(0, tile_size - 1))
    stride = max(1, tile_size - overlap)
    xs = list(range(0, max(1, width), stride))
    ys = list(range(0, max(1, height), stride))
    tiles: list[Tile] = []
    for y in ys:
        for x in xs:
            x2 = min(width, x + tile_size)
            y2 = min(height, y + tile_size)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            tile = Tile(x1, y1, x2, y2)
            if tile not in tiles:
                tiles.append(tile)
    return tiles

