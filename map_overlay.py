"""Plot greedy vs RL placements on a satellite basemap.

The script reads JSON experiment outputs, converts placement coordinates into
GeoDataFrames, and overlays them on Esri WorldImagery tiles inside a fixed
bounding box covering downtown/USC. Markers are rendered with a subtle halo so
both methods remain visible even over satellite imagery.

Usage
-----
python map_overlay.py [--show] [--save out.png]

Dependencies
------------
contextily, geopandas, shapely, matplotlib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box

# Bounding box (lat/lon)
LAT_MIN, LAT_MAX = 33.99, 34.04
LON_MIN, LON_MAX = -118.31, -118.26

# Result files to plot: (filename, label, color, marker), change jsons here
RESULT_FILES: List[Tuple[str, str, str, str]] = [
    ("Approach1_Greedy_k20.json", "Greedy (Approach 2)", "#ff7043", "o"),
    ("Approach1_RL_k20.json", "RL (Approach 2)", "#29b6f6", "^"),
]

RESULT_DIR = Path("new_reward") / "results" / "region_specific" / "All"


def load_points(json_path: Path) -> gpd.GeoDataFrame:
    """Load placement coordinates from a JSON results file."""
    data = json.loads(json_path.read_text())
    coords = data.get("placement_coordinates", [])
    if not coords:
        raise ValueError(f"No coordinates found in {json_path}")

    geometry = [Point(entry["longitude"], entry["latitude"]) for entry in coords]
    gdf = gpd.GeoDataFrame(coords, geometry=geometry, crs="EPSG:4326")
    gdf["method"] = data.get("method", json_path.stem)
    return gdf


def build_plot(datasets: Iterable[gpd.GeoDataFrame], labels: Iterable[str],
               colors: Iterable[str], markers: Iterable[str], show: bool,
               save_path: Path | None) -> None:
    """Plot datasets over the specified bounding box on a satellite basemap."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Convert bounding box to Web Mercator limits
    bounds = gpd.GeoSeries(box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX), crs="EPSG:4326").to_crs(epsg=3857)
    xmin, ymin, xmax, ymax = bounds.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for gdf, label, color, marker in zip(datasets, labels, colors, markers):
        gdf_web = gdf.to_crs(epsg=3857)
        # Halo layer keeps markers legible against imagery
        gdf_web.plot(
            ax=ax,
            marker=marker,
            color="black",
            markersize=120,
            alpha=0.35,
        )
        gdf_web.plot(
            ax=ax,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            markersize=90,
            label=label,
            alpha=0.95,
        )

    ctx.add_basemap(
        ax,
        crs="EPSG:3857",
        source=ctx.providers.Esri.WorldImagery,
        alpha=0.75,
    )
    ax.set_title("Shade Placement Comparison (Approach 1 k = 20)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved map to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot greedy vs RL placements on satellite imagery")
    parser.add_argument("--show", action="store_true", help="Display the plot window")
    parser.add_argument("--save", type=Path, help="Optional path to save the figure as PNG/PDF")
    args = parser.parse_args()

    datasets: List[gpd.GeoDataFrame] = []
    labels: List[str] = []
    colors: List[str] = []
    markers: List[str] = []

    for filename, label, color, marker in RESULT_FILES:
        json_path = RESULT_DIR / filename
        if not json_path.exists():
            raise FileNotFoundError(f"Could not find {json_path}")
        gdf = load_points(json_path)
        datasets.append(gdf)
        labels.append(label)
        colors.append(color)
        markers.append(marker)

    build_plot(datasets, labels, colors, markers, show=args.show, save_path=args.save)


if __name__ == "__main__":
    main()
