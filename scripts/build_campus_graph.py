#!/usr/bin/env python3
"""
Build the campus routing graph from OpenStreetMap. Run on your LAPTOP once
(needs internet + osmnx); copy the output graphml to the Pi. The Pi only
needs networkx to read it (see mycar/parts/gps_nav.py).

Usage:
    pip install osmnx
    python build_campus_graph.py --point 26.4725,73.1140 --dist 800
    python build_campus_graph.py --place "IIT Jodhpur, India"

Then:  scp campus_graph.graphml pi@raspberrypi.local:~/mycar/
"""
import argparse

import osmnx as ox

OUT = "campus_graph.graphml"


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--point", help="lat,lon center of campus")
    g.add_argument("--place", help="OSM place name")
    ap.add_argument("--dist", type=int, default=800,
                    help="radius in meters around --point")
    # 'all' includes footways/paths — the cart drives sidewalks, not just roads
    ap.add_argument("--network-type", default="all")
    args = ap.parse_args()

    if args.point:
        lat, lon = map(float, args.point.split(","))
        G = ox.graph_from_point((lat, lon), dist=args.dist,
                                network_type=args.network_type)
    else:
        G = ox.graph_from_place(args.place, network_type=args.network_type)

    # gps_nav runs plain networkx shortest_path on 'length' edge weights
    G = ox.convert.to_undirected(G)
    ox.save_graphml(G, OUT)
    print(f"Wrote {OUT}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("Copy it to the Pi and set CAMPUS_GRAPHML in myconfig.py")


if __name__ == "__main__":
    main()
