import json
import math

def distance(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def init_track(points_file="points.json", track_file="track.json"):
    # Load points
    with open(points_file, "r") as f:
        points = json.load(f)
    points = [tuple(p) for p in points]

    # Close the loop
    segments = []
    lengths = []
    total_length = 0

    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i+1) % len(points)]
        seg_length = distance(p1, p2)
        segments.append((p1, p2))
        lengths.append(seg_length)
        total_length += seg_length

    # Compute cumulative normalized progression values
    cumulative = [0.0]
    acc = 0.0
    for l in lengths:
        acc += l
        cumulative.append(acc / total_length)

    track_data = {
        "points": points,
        "segments": segments,
        "lengths": lengths,
        "total_length": total_length,
        "cumulative": cumulative
    }

    # Save to track.json
    with open(track_file, "w") as f:
        json.dump(track_data, f, indent=4)

    print(f"Track initialized with {len(points)} points, total length {total_length:.2f}")
    return track_data

if __name__ == "__main__":
    init_track()
