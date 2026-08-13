#!/usr/bin/env python3
"""
Build the campus routing graph from OpenStreetMap. Run on your LAPTOP once
(needs internet + osmnx); copy the output graphml to the EC2 MISSION SERVER,
which does the routing. The Pi never sees it.

WHY NOT GOOGLE DIRECTIONS API
-----------------------------
Google returns an encoded polyline meant for a human following turn-by-turn
directions on public roads. Here we need a GRAPH WE OWN, because a campus cart
has constraints Google knows nothing about:

  * a path with a speed breaker too tall for the chassis -> delete that edge
  * a service road the cart may not use -> delete that edge
  * a shortcut through a building gate that OSM lacks -> add that edge
  * route again after any of the above, offline, in milliseconds

You cannot express "this path is impassable for MY vehicle" to the Directions
API, and campus interior paths are frequently absent from its routing network
anyway. Owning the graphml also means no API key, no per-request billing, and
no internet dependency at dispatch time.

Google imagery is still the better thing to LOOK at while placing pins — the
app uses satellite tiles for exactly that reason. Tiles and routing are
separate decisions.

To prune or extend the graph afterwards, load it with networkx and use
remove_edge / add_edge, then save it back with nx.write_graphml.

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
    print("Copy it to the EC2 mission server:")
    print(f"  scp {OUT} ec2-user@<host>:~/campus-cart/")
    print("  python server/app.py --graph campus_graph.graphml")
    print("\nCheck the coverage before trusting it — OSM often misses campus")
    print("interior footpaths. Compare against the satellite view in the app;")
    print("any path the cart should use but the graph lacks must be added, or")
    print("routing will detour around it.")


if __name__ == "__main__":
    main()
